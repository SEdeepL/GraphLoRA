import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V 的线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # 用于多头输出拼接后的线性变换
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换 Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # 将 Q, K, V 分为 num_heads 个头部
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(Q.device)  # [batch_size, num_heads, seq_len, seq_len]
        
        # 应用 mask（可选）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # 通过 softmax 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 加权求和: attention_weights * V
        out = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 将多头输出拼接起来
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [batch_size, seq_len, embed_dim]
        
        # 线性变换得到最终输出
        out = self.out_linear(out)  # [batch_size, seq_len, embed_dim]
        
        return out
