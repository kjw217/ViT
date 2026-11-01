import model
from torchsummary import summary

vit_small = model.ViT(image_size=224,
                     patch_size=16,
                     layer_n=10,
                     embd_d=512,
                     ffn_dim=2048,
                     head_n=8,
                     cls_n=1000,
                     attn_dropout=0.1)


vit_base = model.ViT(image_size=224,
                     patch_size=16,
                     layer_n=12,
                     embd_d=768,
                     ffn_dim=3072,
                     head_n=12,
                     cls_n=21841,
                     attn_dropout=0.1)

vit_large = model.ViT(image_size=224, 
                     patch_size=16,
                     layer_n=24,
                     embd_d=1024,
                     ffn_dim=4096,
                     head_n=16,     
                     cls_n=21841,
                     attn_dropout=0.1)

if __name__ == "__main__" :
    summary(vit_base, input_size=(3, 384, 384), batch_size = 10, device="cpu")
