import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.size(0), :].to(x.device)
        x = x + pe
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        device = query.device
        batch_size = query.size(0)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(device)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(device)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).to(device)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(attn, value).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(output)

class QFormer(nn.Module):
    def __init__(self, num_heads, num_layers):
        super(QFormer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=512)
        self.self_attn_layers = nn.ModuleList([MultiHeadAttention(d_model=512, num_heads=num_heads) for _ in range(num_layers)])
        self.cross_attn_layers_img = nn.ModuleList([MultiHeadAttention(d_model=512, num_heads=num_heads) for _ in range(num_layers)])
        self.cross_attn_layers_text = nn.ModuleList([MultiHeadAttention(d_model=512, num_heads=num_heads) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)) for _ in range(num_layers)])
        
    def forward(self, image_features, text_features):
        device = image_features.device
        batch_size, d_model_img, height, width = image_features.size()
        num_queries = height * width
        d_model_text = text_features.size(-1)

        if self.pos_encoder.pe.size(-1) != d_model_img:
            self.pos_encoder = PositionalEncoding(d_model=d_model_img).to(device)

        self.self_attn_layers = nn.ModuleList([MultiHeadAttention(d_model=d_model_img, num_heads=self.self_attn_layers[0].num_heads).to(device) for _ in range(len(self.self_attn_layers))])
        self.cross_attn_layers_img = nn.ModuleList([MultiHeadAttention(d_model=d_model_img, num_heads=self.cross_attn_layers_img[0].num_heads).to(device) for _ in range(len(self.cross_attn_layers_img))])
        self.cross_attn_layers_text = nn.ModuleList([MultiHeadAttention(d_model=d_model_img, num_heads=self.cross_attn_layers_text[0].num_heads).to(device) for _ in range(len(self.cross_attn_layers_text))])
        self.ffn_layers = nn.ModuleList([nn.Sequential(nn.Linear(d_model_img, d_model_img).to(device), nn.ReLU(), nn.Linear(d_model_img, d_model_img).to(device)) for _ in range(len(self.ffn_layers))])

        text_to_img_fc = nn.Linear(d_model_text, d_model_img).to(device)
        text_features = text_to_img_fc(text_features.to(device))

        queries = torch.randn(batch_size, num_queries, d_model_img, device=device)

        image_features_flat = image_features.view(batch_size, d_model_img, num_queries).permute(0, 2, 1).to(device)
        
        for self_attn, cross_attn_img, cross_attn_text, ffn in zip(self.self_attn_layers, self.cross_attn_layers_img, self.cross_attn_layers_text, self.ffn_layers):
            queries = self.pos_encoder(queries)
            queries = self_attn(queries, queries, queries)
            queries = cross_attn_img(queries, image_features_flat, image_features_flat)
            queries = cross_attn_text(queries, text_features, text_features)
            queries = ffn(queries)
        
        queries_reshaped = queries.permute(0, 2, 1).view(batch_size, d_model_img, height, width)

        queries_reshaped.to(device)
        
        return queries_reshaped
