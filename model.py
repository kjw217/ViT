import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class mlp_block(nn.Module) :
    def __init__(self, embd_dim, ffn_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ffnA = nn.Linear(embd_dim, ffn_dim)
        self.ffnB = nn.Linear(ffn_dim, embd_dim)
    def forward(self, x) :
        output = F.gelu(self.ffnB(self.ffnA(x)))
        return output


class transformer_block(nn.Module) :
    def __init__(self, image_size, patch_size, embd_dim, head_n, ffn_dim, attn_dropout,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MSA = nn.MultiheadAttention(embed_dim=embd_dim,
                                         num_heads=head_n,
                                         dropout=attn_dropout,
                                         batch_first=True)
        self.MLP = mlp_block(embd_dim=embd_dim,
                             ffn_dim=ffn_dim)
        self.lnA = nn.LayerNorm([(image_size//patch_size)**2 + 1, embd_dim])
        self.lnB = nn.LayerNorm([(image_size//patch_size)**2 + 1, embd_dim])
    def forward(self, x) :
        x = self.lnA(x)
        attn_out, attn_score = self.MSA(x, x, x)
        attn_out = attn_out + x

        mlp_out = self.MLP(self.lnB(attn_out)) + attn_out
        return mlp_out

class ViT(nn.Module) :
    def __init__(self,image_size, embd_d, patch_size ,layer_n, head_n, ffn_dim, attn_dropout, cls_n, *args, **kwargs):
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
                                                                    attn_dropout=attn_dropout) for _ in range(layer_n)])

        self.ln = nn.LayerNorm(embd_d)
        self.mlp_head = nn.Linear(embd_d, cls_n)

    def forward(self, x) :
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
    test_model = ViT(image_size=224, embd_d=128, patch_size=16, layer_n=1, head_n=2, ffn_dim=128, attn_dropout=0.1, cls_n=1)
    x = torch.randn((5, 3, 224, 224))
    print(test_model(x).shape)
