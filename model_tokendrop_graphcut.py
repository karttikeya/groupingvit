import torch
from torch import nn
from math import log2
from utils_ed_plus import Batched_ED_Plus #DC-based ED for dim<64

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA

batched_ed_plus = Batched_ED_Plus.apply

class CIFARViT(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(2, 2,),  # this patch size is used for CIFAR-10
        # --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        keep_eta = 1.0,
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    # keep_eta = keep_eta
                )
                for _ in range(self.depth)
            ]
        )


        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )

        self.head = nn.Linear(self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # patchification using conv and flattening
        # + abolsute positional embeddings
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        for layer in self.layers:
            x = layer(x)

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x)

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x


class Block(nn.Module):


    def __init__(self, dim, num_heads):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """

        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads
        )

        self.G = MLPSubblock(dim=dim)

    def forward(self, x, eta=1.0):
        x_f = self.F(x, eta)
        x = x_f #+ torch.gather(x, 1, indices)
        x = self.G(x) + x
        return x

class MLPSubblock(nn.Module):


    def __init__(
        self, dim, mlp_ratio=4,  # standard for ViTs
    ):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x):
        return self.mlp(self.norm(x))



class AttentionSubBlock(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, max_out_sz=256):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(head_dim, dim)

    def forward(self, x, eta):
        B, N, C = x.shape
        H = self.num_heads
        c = C // H  # dim per each head
        
        # We concat head outputs on the sequence dim (as opposed to embd dim).
        # Hence we divide the num of out tokens by H. The out tokens are set by
        # all possible (+,+,-, ...) combinations of eigen vecs. To make sure the
        # out token num is a power of 2, we first calculate num of eigen pairs
        N_eigs = max(int(log2(N * eta / H)), 1)
        N_out = 2 ** N_eigs

        # chop and proj to query, key, values
        qkv = self.qkv(x).reshape(B, N, 3, H, c).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each [B, H, N, c]

        # calc attention matrix
        attn = (self.scale * q) @ q.transpose(-2, -1)

        # zeroize diag
        mask = torch.eye(*attn.shape[-2:], device=attn.device).bool().unsqueeze(0)
        attn.masked_fill_(mask, -torch.inf)
        attn = attn.softmax(dim=-1) 

        # must be symmetric
        attn = (attn + attn.transpose(-2, -1)) / 2

        # calc graph laplacian matrix, normalize it as in normalized cuts
        deg_vec = attn.sum(-1)
        deg = torch.diag_embed(deg_vec, offset=0, dim1=-2, dim2=-1)
        # We want (deg**(-0.5))*(deg-attn)*(deg**(-0.5))
        # the diagonal matrix has zeros, instead of taking pinv we can do the
        # trick of calculating on the vec
        inv_sqrt_deg_vec = (deg_vec ** (-0.5)).unsqueeze(-1)  # [B, H, N, 1]
        lap = inv_sqrt_deg_vec * (deg - attn) * inv_sqrt_deg_vec.transpose(-2, -1)

        # get K+1 smallest eigen vals and vecs 
        # eig_vals, eig_vecs = torch.lobpcg(lap, k=N_eigs+1, largest=False, method="ortho")
        eig_vecs, eig_diag = batched_ed_plus(lap.view(B*H, N, N))


        # take the values on diagonal and sort them, use indices to sort the vecs too
        eig_vals, ids = torch.diagonal(eig_diag, dim1=-2, dim2=-1).sort(dim=-1)
        eig_vecs = torch.gather(eig_vecs, -2, ids.unsqueeze(-1).expand(eig_vecs.shape)) 

        # First eig vec corresondse to eig val 0, and is a*ones, not useful,
        eig_vecs = eig_vecs[:, 1:N_eigs+1].view(B, H, N_eigs, N)
        
        
        if eig_vecs.isnan().any() or eig_vecs.isinf().any(): # or eig_vals.isnan().any() or eig_vals.isinf().any():
            import ipdb
            ipdb.set_trace()
        # print('vecs', eig_vecs[0, 0, 0])
        # print('vals', eig_vals[0])


        # First eig vec corresondse to eig val 0, and is a*ones, not useful.
        # eig_vecs = eig_vecs.transpose(-2, -1)[:, :, 1:]  # [B, H, N_eigs, N]

        # all possible (+1, -1) combinations with output length -> [N_out, N_eigs]
        combs = torch.cartesian_prod(*([torch.tensor([1., -1], device=x.device)] * N_eigs))
        if combs.dim() == 1:
            combs = combs.unsqueeze(1)  # in case only one eig

        # get modified attention, each row is a group, mutliply by values to get output
        group_attn = torch.einsum('OE,BHEN->BHON', combs, eig_vecs) # [B, H, N_out, N]
        group_attn = group_attn.softmax(-1)
        x = (group_attn @ v)  # [B, H, N_out, c]

        # We concatenate tokens from all heads as more tokens ('spatial/sequence dim') 
        # as opposed to the classic concat on the channels/embd/features dim
        x = x.reshape(B, H*N_out, c) 
        x = self.proj(x)  # proj goes head_dim (c) -> dim (C) since we didn't concat on the chans
        return x








        # eig_vecs_list, eig_diag_list = [], []
        # for h in range(H):
        #     eig_vecs, eig_diag = batched_ed_plus(lap[:, h])
        #     eig_vecs_list.append(eig_vecs)
        #     eig_diag_list.append(eig_diag)
        # eig_vecs = torch.stack(eig_vecs_list, dim=1)
        # eig_diag = torch.stack(eig_diag_list, dim=1)