import torch
from torch import nn
import math

class SelfAttention(nn.Module):
    def __init__(self, emb_dim, num_attn_heads):
        super().__init__()
        self.attn_head_size = emb_dim // num_attn_heads
        self.q_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.k_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        self.v_proj = nn.Linear(in_features=emb_dim, out_features=emb_dim)
        
    def transpose_for_scores(self, tensor):
        tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], self.attn_head_size, -1).contiguous()
        return tensor.transpose(-1, 1).transpose(-1, -2)

    def forward(self, x):
        # Given that we have x of shape B, L, emb_dim (B - batch size, L amount of tokens) 
        # In the next code we get Q, K, V and split the x for multi head attention
        q = self.q_proj(x)
        q = self.transpose_for_scores(q)
        k = self.k_proj(x)
        k = self.transpose_for_scores(k)
        v = self.v_proj(x)
        v = self.transpose_for_scores(v)
        scores = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(self.attn_head_size), dim=-1)
        # 1, 8, 197, 64
        attn_scores = scores @ v
        # Reshape to 1, 197, 8, 64
        attn_scores = attn_scores.transpose(1, 2)
        # 1, 197, 512
        output = attn_scores.reshape(*attn_scores.shape[:2], -1)
        return output

class MLPLayer(nn.Module):
    def __init__(self, emb_dim, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(emb_dim, intermediate_size)
        self.silu = nn.SiLU(inplace=True)
        self.output_proj = nn.Linear(intermediate_size, emb_dim)
    
    def forward(self, hidden_states):
        hidden_states = self.silu(self.dense(hidden_states))
        # Scale it back to original embedding size
        hidden_states = self.output_proj(hidden_states)
        return hidden_states
        

class ViTLayer(nn.Module):
    def __init__(self, emb_dim, num_attn_heads, intermediate_size):
        super().__init__()
        self.norm_before = nn.LayerNorm(emb_dim)
        self.norm_after = nn.LayerNorm(emb_dim)
        self.self_attn = SelfAttention(emb_dim, num_attn_heads)
        self.mlp_layer = MLPLayer(emb_dim, intermediate_size)
    
    def forward(self, hidden_states):
        # Input will be [1, (image_size // patch_size) ** 2 + 1, emb_dim]
        # Normalize before attention
        normalized_hidden_states = self.norm_before(hidden_states)
        attention_output = self.self_attn(normalized_hidden_states)
        # residual connection
        hidden_states = hidden_states + attention_output
        # Normalize after attention and before MLP
        normalized_hidden_states = self.norm_after(hidden_states)
        # MLP Layer, outputs shape of [1, 197, intermediate_size] which needs to get 
        # scaled back to the emb_dim size
        mlp_output = self.mlp_layer(normalized_hidden_states)
        # 2nd residual after mlp
        return hidden_states + mlp_output

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_attn_heads, intermediate_size, num_layers):
        super().__init__()
        self.vit_layer = nn.ModuleList([ViTLayer(emb_dim, num_attn_heads, intermediate_size) for _ in range(num_layers)])
    
    def forward(self, hidden_states):
        for module in self.vit_layer:
            hidden_states = module(hidden_states)
        return hidden_states


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_attn_heads, intermediate_size, num_layers, num_registers):
        super().__init__()
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.pos_embeddings = nn.Parameter(torch.rand((1, (image_size // patch_size) ** 2 + 1 + num_registers, emb_dim))) # + 1 for cls token
        nn.init.xavier_uniform_(self.pos_embeddings)
        self.conv = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.rand((1, 1, emb_dim)))
        self.register_tokens = nn.Parameter(torch.rand((1, num_registers, emb_dim)))
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.register_tokens)
        self.encoder = TransformerEncoder(emb_dim, num_attn_heads, intermediate_size, num_layers)
    
    def flatten_image(self, image: torch.Tensor):
        """
        Flattens the image to represent it as tokens for transformer.
        Supports only square images.

        image: B, C, H, W tensor
        patch_size: int
        """
        x = self.conv(image)
        x = x.flatten(2) # (B,D,P,P) -> (B, D, P*P) (P being patch size and D being embedding dim)
        x = x.transpose(1, 2) # (B, P*P, D)
        return x
    
    def forward(self, x):
        x = self.flatten_image(x) # output shape of B, P*P, D
        x = torch.cat([x, self.cls_token.expand(x.shape[0], -1, -1), self.register_tokens.expand(x.shape[0], -1, -1)], dim=-2) # B, P*P+1, D
        x += self.pos_embeddings
        x = self.encoder(x)
        return x


def main():
    image = torch.rand((16, 3, 224, 224)).to("cuda:0")
    vit_transformer = VisionTransformer(patch_size=16, emb_dim=512, image_size=224,num_attn_heads=8, intermediate_size=768, num_layers=4, num_registers=64)
    vit_transformer.to("cuda:0")
    with torch.inference_mode():
        out = vit_transformer(image)
    print(out.shape)

if __name__ == "__main__":
    main()
