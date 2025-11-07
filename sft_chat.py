import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
from transformers import BertTokenizerFast


# ======================
# 配置
# ======================
class InferConfig:
    vocab_path = "bert-base-chinese"
    model_path = r"E:\deep-learning\ml\minimind_large\step_60000.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 512
    temperature = 0.8
    top_k = 90
    top_p = 0.85


cfg = InferConfig()


# ======================
# MiniMind 模型结构（保持训练时一致）
# ======================
class MiniMindBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)

        # SwiGLU 前馈
        hidden_dim = dim * 4
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.GLU(),  # 代替 GELU
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-LN Transformer
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x


class MiniMind(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            MiniMindBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # 权重共享

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)

        # Causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)

        x = self.ln(x)
        logits = self.head(x)
        return logits  # (B, L, V)


# ======================
# 加载模型和 tokenizer
# ======================
tokenizer = BertTokenizerFast.from_pretrained(cfg.vocab_path)
special_tokens = ["<|im_start|>", "<|im_end|>", "<|user|>", "<|assistant|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
vocab_size = len(tokenizer)

model = MiniMind(
    vocab_size=vocab_size,
    hidden_dim=512,
    n_layers=24,
    n_heads=8,
    max_len=cfg.max_len
).to(cfg.device)

model.load_state_dict(torch.load(cfg.model_path, map_location=cfg.device))
model.head.weight = model.embed.weight
model.eval()


# ======================
# 推理函数
# ======================
def generate(text_prompt, max_gen_len=512, repetition_penalty=1.1):
    """
    text_prompt: 用户输入，已经带 <|user|> 标签
    max_gen_len: 最多生成多少 token
    repetition_penalty: 重复惩罚系数 (>1 会降低重复 token 的概率)
    """
    input_text = f"<|im_start|><|user|>{text_prompt}<|im_end|><|im_start|><|assistant|>"
    input_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    input_ids = input_ids[-cfg.max_len:]  # 保证长度不超过 max_len
    input_ids = torch.tensor([input_ids], device=cfg.device)

    for _ in range(max_gen_len):
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        logits = model(input_ids, attention_mask=attention_mask)
        next_token_logits = logits[0, -1, :] / cfg.temperature

        # ====== 添加重复惩罚 ======
        for token_id in set(input_ids[0].tolist()):
            if next_token_logits[token_id] > 0:
                next_token_logits[token_id] /= repetition_penalty
            else:
                next_token_logits[token_id] *= repetition_penalty

        # Top-k + Top-p sampling
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=cfg.top_k, top_p=cfg.top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # 遇到 <|im_end|> 就停止
        if next_token.item() == tokenizer.convert_tokens_to_ids("<|im_end|>"):
            break

    output_ids = input_ids[0].tolist()
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    if "<|assistant|>" in output_text:
        output_text = output_text.split("<|assistant|>")[-1].strip()
    return output_text



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """Top-K + Top-P (nucleus) 采样"""
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # 保留第一个超过 top_p 的 token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits





if __name__ == "__main__":
    prompt = "我感到焦虑该怎么办？"
    response = generate(prompt, max_gen_len=512)
    print(prompt)
    print(" Assistant:", response)
