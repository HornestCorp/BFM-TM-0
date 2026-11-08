import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embd):
        super().__init__()

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)

        scale = self.head_size ** -0.5
        wei = wei * scale

        tril = self.tril[:T, :T]
        wei = wei.masked_fill(tril == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v 

        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, n_embd, block_size):
        super().__init__()

        self.proj = nn.Linear(n_embd, n_embd)

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([Head(head_size=head_size, block_size=block_size, n_embd=n_embd) for _ in range(num_heads)])

    def forward(self, x):
        heads_check = [h(x) for h in self.heads]

        multi_heads = torch.cat(heads_check, dim=-1)

        end_heads = self.proj(multi_heads)

        return end_heads

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd) ,nn.ReLU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(num_heads=num_heads, block_size=block_size, n_embd=n_embd)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 

class BFMramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, num_heads):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.ln_f = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[Block(num_heads=num_heads, n_embd=n_embd, block_size=block_size) for _ in range(n_layer)])

        self.block_size = block_size

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)

        B, T = idx.shape
        pos_T = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos_T)

        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)

        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
