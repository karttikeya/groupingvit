import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

# We use the standard pytorch multi-head attention module
from torch.nn import MultiheadAttention as MHA


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
                    keep_eta = keep_eta
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


    def __init__(self, dim, num_heads, keep_eta):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """

        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, keep_eta = keep_eta
        )

        self.G = MLPSubblock(dim=dim)

    def forward(self, x):
        
        x_f, indices = self.F(x)
        x = x_f + torch.gather(x, 1, indices)
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

    def __init__(
        self, dim, num_heads, keep_eta 
    ):

        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MHA(dim, num_heads, 
        batch_first=True)
        self.keep_eta = keep_eta

    def forward(self, x):

        x = self.norm(x)

        # just so as to be able to use the standard pytorch MHA module
        # with torch.no_grad():

        out, weights = self.attn(x, x, x, need_weights=True)

        # sum across dim=2 is 1. 
        weights = weights.sum(dim = 1)
        k = int(self.keep_eta * weights.shape[1])

        import ipdb; ipdb.set_trace()
        _, topk_indices = torch.topk(weights, k, dim = 1, largest = True, sorted = False)

        # index out with topk_indices along dimension 0 and 1 
        # and return the indixed values 
        topk_indices = topk_indices.unsqueeze(2).repeat(1, 1, x.shape[2])
        out = torch.gather(out, 1, topk_indices)
        return out, topk_indices

