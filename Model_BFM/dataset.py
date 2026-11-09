import torch
from pathlib import Path
from .confige import block_size, batch_size

# - FILE: text.txt -
BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "text.txt"
text = DATA_PATH.read_text(encoding='utf-8')

# - DEVICE
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - DICTIONARY and SORTING
char =  sorted(set(text))

stoi = {ch: i for i, ch in enumerate(char)}
itos = {i: ch for i, ch in enumerate(char)}

# ~ List Number
def encode(s):
    return [stoi[c] for c in s]

# ~ Line
def decode(lst):
    return ''.join([itos[i] for i in lst])

# - ADD TENSOR
data = torch.tensor(encode(text))

n = int(0.9 * len(data))

train_data = data[:n]
val_data = data[n:]

vocab_size = len(char)

# ~ Train or Val
def get_batch(split):
    data_src = train_data if split == 'train' else val_data
    top_line = len(data_src) - block_size

    ix = torch.randint(top_line, (batch_size,))

    stack_x = torch.stack([data_src[i:i+block_size] for i in ix])
    stack_y = torch.stack([data_src[i+1:i+block_size+1] for i in ix])

    more_x = stack_x.to(device)
    more_y = stack_y.to(device)

    return more_x, more_y
