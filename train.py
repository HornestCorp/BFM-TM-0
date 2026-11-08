import torch
from dataset import get_batch, vocab_size, decode, device, char
from confige import  block_size, n_embd, n_layer, num_heads, max_range, batch_size
from model import BFMramLanguageModel
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "bfm-tm-0.pth"

model = BFMramLanguageModel(vocab_size, n_embd, block_size, n_layer, num_heads)
model.to(device)
optimization = torch.optim.AdamW(model.parameters(), lr=1e-3)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

for step in range(max_range):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimization.zero_grad(set_to_none=True)
    loss.backward()
    optimization.step()

    if step % 650 == 0: 
       losses = estimate_loss()

       print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=500)

torch.save({
    'model_state_dict': model.state_dict(),
    'n_embd': n_embd,
    'n_layer': n_layer,
    'num_heads': num_heads,
    'batch_size': batch_size,
    'block_size': block_size,
    'max_range': max_range,
    'char': char
}, "bfm-tm-0.pth")

print(decode(out[0].tolist()))
print('')
print(sum(p.numel() for p in model.parameters()))
