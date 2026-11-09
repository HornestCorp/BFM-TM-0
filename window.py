import torch
from pathlib import Path
from Model_BFM import model
from Model_BFM import dataset

MODEL_DIR = Path(__file__).parent

JACOB = MODEL_DIR / "Model_BFM" / "Models" / "bfm-tm-0_1.pth"
LUCI = MODEL_DIR / "Model_BFM" / "Models" / "bfm-tm-0_1-2.pth"
ADAM = MODEL_DIR / "Model_BFM" / "Models" / "bfm-tm-0_1-3.pth"

def print_help():
    print("Enter command for start:")
    print("Start - Command to initiate model generation (run once)")
    print("Help - All command")
    print("Exit - End the program\n")

def print_help_v():
    print("Select a Model:")
    print("1 - Jacob(Low)")
    print("2 - Lusi(Medium)")
    print("3 - Adam(Heavy)")
    print("4 - Help for Version")
    print("5 - Exit")

def error_input():
    print("Enter Help, to see everything.\n")

model_load = False

print("=== Model BFM-TM-0.1 ===\n")
print_help()

while True:
    try:
        usr = input("User: ").strip().lower()
        print("")
    except(KeyboardInterrupt, EOFError):
        print("You have entered incorrect data")
        error_input()

        continue

    if usr == 'start':

        print_help_v()

        while True:
            try: 
                usr_ver = int(input("User-V: "))
                print("")

            except(KeyboardInterrupt, EOFError):
                print("You have entered incorrect data")
                error_input()

                continue

            except ValueError:
                print(f"Not Integer!")
                error_input()

                continue

            if usr_ver == 1:
                check_jacob = torch.load(JACOB, map_location="cpu")

                vocab_size = len(check_jacob['char'])
                n_embd = check_jacob['n_embd']
                block_size = check_jacob['block_size']
                n_layer = check_jacob['n_layer']
                num_heads = check_jacob['num_heads']
                dropout = check_jacob['dropout']
                remeber = check_jacob['model_state_dict']

                model_v = model.BFMramLanguageModel(vocab_size, n_embd, block_size, n_layer, num_heads, dropout)
                model_v.load_state_dict(remeber)
                model_v.eval()

                idx = torch.zeros((1, 1), dtype=torch.long, device="cpu")
                out = model_v.generate(idx, max_new_tokens=200)

                print(dataset.decode(out[0].tolist()), "\n")
                break

            elif usr_ver == 2:
                check_luci = torch.load(LUCI, map_location="cpu")
                
                vocab_size = len(check_luci['char'])
                n_embd = check_luci['n_embd']
                block_size = check_luci['block_size']
                n_layer = check_luci['n_layer']
                num_heads = check_luci['num_heads']
                dropout = check_luci['dropout']
                remeber = check_luci['model_state_dict']
                
                model_v = model.BFMramLanguageModel(vocab_size, n_embd, block_size, n_layer, num_heads, dropout)
                model_v.load_state_dict(remeber)
                model_v.eval()
                
                idx = torch.zeros((1, 1), dtype=torch.long, device="cpu")
                out = model_v.generate(idx, max_new_tokens=200)
                
                print(dataset.decode(out[0].tolist()), "\n")

                break

            elif usr_ver == 3:
                check_adam = torch.load(ADAM, map_location="cpu")
                
                vocab_size = len(check_adam['char'])
                n_embd = check_adam['n_embd']
                block_size = check_adam['block_size']
                n_layer = check_adam['n_layer']
                num_heads = check_adam['num_heads']
                dropout = check_adam['dropout']
                remeber = check_adam['model_state_dict']
                
                model_v = model.BFMramLanguageModel(vocab_size, n_embd, block_size, n_layer, num_heads, dropout)
                model_v.load_state_dict(remeber)
                model_v.eval()
                
                idx = torch.zeros((1, 1), dtype=torch.long, device="cpu")
                out = model_v.generate(idx, max_new_tokens=200)
                
                print(dataset.decode(out[0].tolist()), "\n")

                break

            elif usr_ver == 4:
                print_help_v()

            elif usr_ver == 5:
                print("End Version!")
                break

            else:
                print("Command not found.") 
                error_input()

    elif usr == 'help':
        print_help()
    elif usr == 'exit':
        print("Code end!")
        break
    else:
        print("Command not found.")
        error_input()
