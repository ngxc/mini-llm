import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from tqdm import tqdm

class Config:
    vocab_path = "bert-base-chinese"
    data_path = r"pretrain_hq.jsonl"
    save_dir = "./minillm_large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 512
    batch_size = 32
    lr = 4e-5
    epochs = 5
    hidden_dim = 1024
    n_layers = 12
    n_heads = 16
    dropout = 0.1
    ratio = 0.1
    warmup_ratio = 0.02
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    checkpoint = None

cfg = Config()

class TextDataset(Dataset):
    def __init__(self, path, tokenizer, max_len, ratio=1.0):
        with open(path, "r", encoding="utf-8") as f:
            self.data = [json.loads(line) for line in f]
        cut = int(len(self.data) * ratio)
        self.data = self.data[:cut]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i]["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return input_ids, attention_mask, labels

class MiniLLMBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        hidden_dim = dim * 4
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.GLU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask)[0]
        x = x + self.ff(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers, n_heads, max_len=512, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            MiniLLMBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None, labels=None):
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        for blk in self.layers:
            x = blk(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        return {"loss": loss, "logits": logits}

def train():
    os.makedirs(cfg.save_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained(cfg.vocab_path)
    tokenizer.add_special_tokens({"additional_special_tokens": cfg.special_tokens})
    vocab_size = len(tokenizer)
    model = MiniLLM(
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.max_len,
        dropout=cfg.dropout
    ).to(cfg.device)
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        state_dict = torch.load(cfg.checkpoint, map_location=cfg.device)
        model.load_state_dict(state_dict, strict=False)
    model.head.weight = model.embed.weight
    dataset = TextDataset(cfg.data_path, tokenizer, cfg.max_len, ratio=cfg.ratio)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    num_training_steps = len(dataloader) * cfg.epochs
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
        for batch in pbar:
            input_ids, attention_mask, labels = [x.to(cfg.device) for x in batch]
            out = model(input_ids, attention_mask, labels)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Epoch {epoch + 1} 平均Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), os.path.join(cfg.save_dir, f"epoch{epoch + 1}.pt"))
    print("训练完成 模型已保存到", cfg.save_dir)

if __name__ == "__main__":
    train()