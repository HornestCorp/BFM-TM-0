import torch, time
from dataset import get_batch, vocab_size, decode, device, char
from confige import  block_size, n_embd, n_layer, num_heads, max_range, batch_size, dropout
from model import BFMramLanguageModel
from pathlib import Path

model = BFMramLanguageModel(vocab_size, n_embd, block_size, n_layer, num_heads, dropout)
model.to(device)
optimization = torch.optim.AdamW(model.parameters(), lr=1e-3)

best_val = float('inf')
t0 = time.time()

MODELS_DIR = Path(__file__).parent / "Models"
MODELS_DIR.mkdir(exist_ok=True)
CHECKPOINT_PATH = MODELS_DIR / "bfm-tm-0_1-3.pth"

assert n_embd % num_heads == 0

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}

    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(X, Y)

            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

for step in range(max_range):
    xb, yb = get_batch('train')
    optimization.zero_grad(set_to_none=True)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(xb, yb)
    loss.backward()

    optimization.step()

    if step % 1000 == 0:
       losses = estimate_loss()

       elapsed = time.time() - t0
       
       print(f"\nTime:{elapsed:.4f}")
       print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}\n")


       if losses['val'].item() < best_val:
           best_val = losses['val'].item()
           
           torch.save({
                'model_state_dict': model.state_dict(),
                'n_embd': n_embd,
                'n_layer': n_layer,
                'num_heads': num_heads,
                'batch_size': batch_size,
                'block_size': block_size,
                'max_range': max_range,
                'char': char,
                'dropout':dropout,
                'val_loss':losses['val'].item(),
                'step': step,

                'name': 'Jacob-v2',
                'size': 'Low',
                'version': 'bfm-tm-0_2'
            }, CHECKPOINT_PATH)
           

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])    

model.eval()
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(idx, max_new_tokens=500)

print(decode(out[0].tolist()))
print('')
print(sum(p.numel() for p in model.parameters()))
