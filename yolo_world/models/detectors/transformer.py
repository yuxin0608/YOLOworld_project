import torch
import torch.nn as nn

class MultiHeadTransformer(nn.Module):
    def __init__(self, num_heads=8):
        super(MultiHeadTransformer, self).__init__()
        self.num_heads = num_heads
        self.transformers = nn.ModuleList()

    def forward(self, img_feats):
        transformed_feats = []

        for i, feat in enumerate(img_feats):
            input_dim = feat.size(-1)
            if len(self.transformers) <= i:
                transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=self.num_heads).to(feat.device)
                self.transformers.append(transformer_layer)

            # Apply transformer
            seq_length = feat.size(1) * feat.size(2)
            projected_feat = feat.view(feat.size(0), seq_length, input_dim).permute(1, 0, 2)  # (seq_len, batch, embed_dim)
            attn_output = self.transformers[i](projected_feat)
            attn_output = attn_output.permute(1, 0, 2).view(feat.size())  # (batch, seq_len, embed_dim) -> (batch, height, width, embed_dim)

            transformed_feats.append(attn_output)

        return tuple(transformed_feats)