import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class mlp_block(nn.Module) :
    def __init__(self, embd_dim, ffn_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffnA = nn.Linear(embd_dim, ffn_dim)
        self.ffnB = nn.Linear(ffn_dim, embd_dim)
    def forward(self, x) :
        output = F.gelu(self.ffnB(self.ffnA(x)))
        return output

class MultiheadSelfAttention(nn.Module) :
    def __init__(self, embed_dim, num_heads, dropout, use_exp_softmax=True, batch_first=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_n = num_heads
        self.scale = (embed_dim//num_heads)**0.5
        self.with_exp_softmax = use_exp_softmax
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * (embed_dim))
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(p=dropout)
    
    def forward(self, x) :
        b, n, d = x.shape

        q, k, v = torch.tensor_split(self.qkv_proj(x), 3, dim=2)
        
        #h개의 n*b 행렬을 각각 곱하게 만듬
        q, k, v = self.split(q), self.split(k), self.split(v)
        attn_score = self.softmax((q @ k.transpose(2, 3)).div(self.scale), dim=-1, with_exp=self.with_exp_softmax)

        output = (attn_score @ v).transpose(1, 2).contiguous().view(b, n, d)
        output = self.output_proj(output)

        return output, attn_score

    def split(self, x:torch.Tensor) :
        b, n, d = x.shape
        return x.view(b, n, self.head_n, -1).transpose(1, 2)

    def softmax(self, x:torch.Tensor, dim, with_exp) :
        if with_exp :
            return x.softmax(dim=dim)
        else :
            return x/x.sum(dim, keepdim=True)

class transformer_block(nn.Module) :
    def __init__(self, image_size, patch_size, embd_dim, head_n, ffn_dim, attn_dropout, use_exp_softmax,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MSA = MultiheadSelfAttention(embed_dim=embd_dim,
                                         num_heads=head_n,
                                         dropout=attn_dropout,
                                         batch_first=True,
                                         use_exp_softmax=use_exp_softmax)
        self.MLP = mlp_block(embd_dim=embd_dim,
                             ffn_dim=ffn_dim)
        self.lnA = nn.LayerNorm([(image_size//patch_size)**2 + 1, embd_dim])
        self.lnB = nn.LayerNorm([(image_size//patch_size)**2 + 1, embd_dim])

        self.attn_score = None
    def forward(self, x) :
        x = self.lnA(x)
        attn_out, attn_score = self.MSA(x)
        attn_out = attn_out + x

        self.attn_score = attn_score

        mlp_out = self.MLP(self.lnB(attn_out)) + attn_out
        return mlp_out

class ViT(nn.Module) :
    def __init__(self,image_size, embd_d, patch_size ,layer_n, head_n, ffn_dim, attn_dropout, cls_n, use_exp_softmax=True ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embd_d = embd_d
        self.patch_size = patch_size

        self.p_embedding = nn.Conv2d(in_channels=3, 
                                out_channels=embd_d, 
                                kernel_size=patch_size,
                                stride=patch_size)
        self.positional_embed = nn.Parameter(torch.randn((image_size//patch_size)**2 + 1, embd_d))
        self.cls_token = nn.Parameter(torch.randn(embd_d))

        self.transformer_encoder = nn.ModuleList([transformer_block(image_size=image_size,
                                                                    patch_size=patch_size,
                                                                    embd_dim=embd_d,
                                                                    head_n=head_n,
                                                                    ffn_dim=ffn_dim,
                                                                    attn_dropout=attn_dropout,
                                                                    use_exp_softmax=use_exp_softmax) for _ in range(layer_n)])

        self.ln = nn.LayerNorm(embd_d)
        self.mlp_head = nn.Linear(embd_d, cls_n)

    def forward(self, x:torch.Tensor) :
        if len(x.shape) < 4 :
            x.unsqueeze(0)
        b, c, h, w = x.shape
        patch_tokens = self.p_embedding(x) # b, d, h/p, w/p
        tokens = torch.cat((self.cls_token.repeat(b, 1, 1), patch_tokens.reshape(b, (h//self.patch_size)**2, self.embd_d)), dim=1)
        tokens = tokens + self.positional_embed

        
        for layer in self.transformer_encoder :
            attn_output = layer(tokens)

        output = self.ln(attn_output[:, 0, :].squeeze())
        output = self.mlp_head(output)
        
        return output

if __name__ == "__main__" :
    vit_tiny = ViT(image_size=384,
                     patch_size=16,
                     layer_n=10,
                     embd_d=512,
                     ffn_dim=2048,
                     head_n=8,
                     cls_n=1000,
                     attn_dropout=0.1,
                     use_exp_softmax=True)
    x = torch.randn((5, 3, 384, 384))
    print(vit_tiny(x).shape)
