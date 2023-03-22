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
        use_mini=True
    ):

        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
       

        if use_mini:
            self.patch_size = patch_size
            num_patches = (image_size[0] // self.patch_size[0]) * (
                image_size[1] // self.patch_size[1]
            )


            self.pos_embeddings = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim)
            )

        self.use_mini = use_mini
        mini_tf = None
        if use_mini:
            mini_tf = CIFARViT(use_mini=False, embed_dim=num_patches, n_head=1, depth=1, keep_eta=keep_eta)


            # Standard Patchification and absolute positional embeddings as in ViT
            self.patch_embed = nn.Conv2d(
                3, self.embed_dim, kernel_size=patch_size, stride=patch_size
            )



            self.head = nn.Linear(self.embed_dim, num_classes, bias=True)
            self.norm = nn.LayerNorm(self.embed_dim)

        self.layers = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    keep_eta = keep_eta,
                    mini_tf = mini_tf
                )
                for _ in range(self.depth)
            ]
        )

        

    def forward(self, x):
        # patchification using conv and flattening
        # + abolsute positional embeddings
        if self.use_mini:
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            x += self.pos_embeddings

        for layer in self.layers:
            x = layer(x)

        if self.use_mini:
            # aggregate across sequence length
            x = x.mean(1)

            # head pre-norm
            x = self.norm(x)

            # pre-softmax logits
            x = self.head(x)

        # return pre-softmax logits
        return x


class Block(nn.Module):


    def __init__(self, dim, num_heads, keep_eta, mini_tf):
        """
        Block is composed entirely of function F (Attention
        sub-block) and G (MLP sub-block) including layernorm.
        """

        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        self.F = AttentionSubBlock(
            dim=dim, num_heads=num_heads, keep_eta = keep_eta, mini_tf=mini_tf
        )

        self.G = MLPSubblock(dim=dim)

    def forward(self, x):
        
        x_f  = self.F(x)
        x = self.F.handle_grouping(x)

        x = x_f + x
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
        self, dim, num_heads, keep_eta, mini_tf
    ):

        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MHA(dim, num_heads, 
        batch_first=True)
        self.keep_eta = keep_eta
        self.H = num_heads
        self.use_mini = True
        if mini_tf is None:
            self.use_mini = False
            self.proj_vals = nn.Linear(dim, int(keep_eta*dim))
        self.mini_tf = mini_tf
    
    def handle_grouping(self, x):
        if not self.use_mini:
            return self.proj_vals(x)
        
        # B, L, C = x.shape
        # #TODO: mix heads after baby transformer
        # x = torch.einsum("BLHC,BHLM->BMHC",x.reshape(B, L, self.H, C // self.H), self.weights)
        # x = x.reshape(B, int(L*self.keep_eta), C)

        return x

    def forward(self, x):

        x = self.norm(x)


        if not self.use_mini:
            x_v = self.handle_grouping(x)
            out, _ = self.attn(x, x, x_v, need_weights=False)
        else:
            B, L, C = x.shape

            out, weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)

            #TODO: stale info warning!
            # weights is [N,num_heads,L,L]
            weights_new = self.mini_tf(weights.flatten(0,1)).view(B, self.H, L, int(L*self.keep_eta))
            weights_new = weights_new.softmax(-2)
            self.weights = weights_new.detach()

            
            #TODO: mix heads after baby transformer
            out = torch.einsum("BLHC,BHLM->BMHC",out.reshape(B, L, self.H, C // self.H), self.weights)
            out = out.reshape(B, int(L*self.keep_eta), C)
            
        return out

