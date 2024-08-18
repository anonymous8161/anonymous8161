import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import copy
import xformers.ops as xops

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class GeGLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.gelu(x) * gate
    
class EfficientSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_output = xops.memory_efficient_attention(q, k, v, scale=self.scale)
        
        x = attn_output.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x
    
class TongueNet(nn.Module):
    def __init__(self, num_classes=8, num_attn_layers=1, hidden_dim=256):
        super(TongueNet, self).__init__()
        
        resnet_whole = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_body = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_edge = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.whole_shared = nn.Sequential(*list(resnet_whole.children())[:7]) 
        self.body_shared = nn.Sequential(*list(resnet_body.children())[:8])  
        self.edge_shared = nn.Sequential(*list(resnet_edge.children())[:8])  
        
        last_stage = list(resnet_whole.children())[7]
        
        self.whole_fur_block = copy.deepcopy(last_stage)
        self.whole_color_block = copy.deepcopy(last_stage)
        
        self.body_shape = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Crack
        )
        
        self.whole_color = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # White, Yellow, Black, Red
        )
        
        self.edge_shape =  copy.deepcopy(self.body_shape) # Toothmark
        
        self.whole_fur = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Yes/No
        )

        self.body_combine = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # RedSpot, Furthick, FurYellow
        )
        self.edge_combine = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # Pale, tipsideread, Ecchymosis
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.attn_norm1 = nn.LayerNorm(2048)
        self.attn_norm2 = nn.LayerNorm(2048)
        self.ffn_norm1 = nn.LayerNorm(2048)
        self.ffn_norm2 = nn.LayerNorm(2048)

        # Attention layers
        # self.cat_edge_attention = SelfAttention(embed_size=2048, heads=8)
        # self.cat_body_attention = SelfAttention(embed_size=2048, heads=8)
        # self.cat_edge_attention = EfficientSelfAttention(dim=2048, num_heads=8)
        # self.cat_body_attention = EfficientSelfAttention(dim=2048, num_heads=8)


        # # feed-forward network after attention
        # self.ffn1 = nn.Sequential(
        #     GeGLU(2048, 1024),
        #     nn.Linear(1024, 2048)
        # )

        # self.ffn2 = nn.Sequential(
        #     GeGLU(2048, 1024),
        #     nn.Linear(1024, 2048)
        # )

        # # Gate fusion
        # self.gate = nn.Parameter(torch.tensor([0.5]))
        
        self.edge_attn_layers = nn.ModuleList([EfficientSelfAttention(dim=2048, num_heads=8) for _ in range(num_attn_layers)])
        self.body_attn_layers = nn.ModuleList([EfficientSelfAttention(dim=2048, num_heads=8) for _ in range(num_attn_layers)])
        
        self.edge_ffn_layers = nn.ModuleList([
            nn.Sequential(
                GeGLU(2048, 1024),
                nn.Linear(1024, 2048)
            ) for _ in range(num_attn_layers)
        ])
        
        self.body_ffn_layers = nn.ModuleList([
            nn.Sequential(
                GeGLU(2048, 1024),
                nn.Linear(1024, 2048)
            ) for _ in range(num_attn_layers)
        ])
        
        self.edge_attn_norms = nn.ModuleList([nn.LayerNorm(2048) for _ in range(num_attn_layers)])
        self.body_attn_norms = nn.ModuleList([nn.LayerNorm(2048) for _ in range(num_attn_layers)])
        self.edge_ffn_norms = nn.ModuleList([nn.LayerNorm(2048) for _ in range(num_attn_layers)])
        self.body_ffn_norms = nn.ModuleList([nn.LayerNorm(2048) for _ in range(num_attn_layers)])

    def clone_block(self, block):
        clone = type(block)()
        clone.load_state_dict(block.state_dict())
        return clone

    def forward(self, whole_img, body_img, edge_img):

        # Extract shared features
        whole_features = self.whole_shared(whole_img)
        body_features = self.body_shared(body_img)
        edge_features = self.edge_shared(edge_img)
        
        # Branch-specific features   
        color_features = self.whole_color_block(whole_features)
        fur_features = self.whole_fur_block(whole_features)

        # Branch predictions
        crack = self.body_shape(body_features)
        toothmark = self.edge_shape(edge_features)
        color = self.whole_color(color_features)
        fur = self.whole_fur(fur_features)
        
        # Prepare features for attention
        # fur_feat = fur_features.mean(dim=[2,3]).unsqueeze(1)
        # edge_color_feat = edge_color_features.mean(dim=[2,3]).unsqueeze(1)
        # body_feat = body_shape_features.mean(dim=[2,3]).unsqueeze(1)
        fur_feat = self.avgpool(fur_features).squeeze(2).squeeze(2).unsqueeze(1)  # 1, 1 , c
        color_feat = self.avgpool(color_features).squeeze(2).squeeze(2).unsqueeze(1)
        edge_feat = self.avgpool(edge_features).squeeze(2).squeeze(2).unsqueeze(1)
        body_feat = self.avgpool(body_features).squeeze(2).squeeze(2).unsqueeze(1)
        
        # Apply attention with LayerNorm and residual connection
        cat_edge_input = torch.cat([fur_feat, color_feat, edge_feat], dim=1)
        cat_body_input = torch.cat([fur_feat, color_feat, body_feat], dim=1)

        # cat_edge_norm = self.attn_norm1(cat_edge_input)
        # cat_body_norm = self.attn_norm2(cat_body_input)

        # cat_edge_att = self.cat_edge_attention(cat_edge_norm) + cat_edge_input
        # cat_body_att = self.cat_body_attention(cat_body_norm) + cat_body_input

        # cat_edge_norm = self.ffn_norm1(cat_edge_att)
        # cat_body_norm = self.ffn_norm2(cat_body_att)

        # cat_edge_att = self.ffn1(cat_edge_norm) + cat_edge_att
        # cat_body_att = self.ffn2(cat_body_norm) + cat_body_att

        cat_edge_att = cat_edge_input
        cat_body_att = cat_body_input

        for i in range(len(self.edge_attn_layers)):
            # Edge attention
            cat_edge_norm = self.edge_attn_norms[i](cat_edge_att)
            cat_edge_att = self.edge_attn_layers[i](cat_edge_norm) + cat_edge_att
            cat_edge_norm = self.edge_ffn_norms[i](cat_edge_att)
            cat_edge_att = self.edge_ffn_layers[i](cat_edge_norm) + cat_edge_att

            # Body attention
            cat_body_norm = self.body_attn_norms[i](cat_body_att)
            cat_body_att = self.body_attn_layers[i](cat_body_norm) + cat_body_att
            cat_body_norm = self.body_ffn_norms[i](cat_body_att)
            cat_body_att = self.body_ffn_layers[i](cat_body_norm) + cat_body_att
        
        # # Final classification
        edge_ensemble_prediction = self.edge_combine(cat_edge_att.mean(dim=1)) # Pale, TipSideRed, Ecchymosis, 
        body_ensemble_prediction = self.body_combine(cat_body_att.mean(dim=1)) # RedSpot, Furthick, FurYellow

        return {
            'crack': torch.sigmoid(crack),
            'toothmark': torch.sigmoid(toothmark),
            'color': torch.sigmoid(color),
            'fur': torch.sigmoid(fur),
            'edge_ensemble_prediction': torch.sigmoid(edge_ensemble_prediction),
            'body_ensemble_prediction': torch.sigmoid(body_ensemble_prediction),
        }
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TongueNet().to(device)
    
    batch_size = 1
    whole_img = torch.randn(batch_size, 3, 256, 256).to(device)
    body_img = torch.randn(batch_size, 3, 256, 256).to(device)
    edge_img = torch.randn(batch_size, 3, 256, 256).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(whole_img, body_img, edge_img)

if __name__ == "__main__":
    main()