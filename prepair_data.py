from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

BOOK1_PATH = DATA_DIR / "book1.txt"
BOOK2_PATH = DATA_DIR / "book2.txt"
TEXT_PATH = DATA_DIR / "text.txt"

text_1 = BOOK1_PATH.read_text(encoding="utf-8")
text_2 = BOOK2_PATH.read_text(encoding="utf-8")

def strip_gutenberg(text):
    position_start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    position_end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")

    position_line_end = text.find("\n", position_start)

    if position_start == -1 or position_end == -1 or position_line_end == -1:
        raise ValueError("Gutenberg markers not found")
    else:    
        main_text = text[position_line_end+1:position_end]

    return main_text

main_text_1 = strip_gutenberg(text_1)
main_text_2 = strip_gutenberg(text_2)

main_text = main_text_1 + "\n\n" + main_text_2

TEXT_PATH.write_text(main_text, encoding="utf-8")

print(len(main_text))