from pathlib import Path

BASE_DIR = Path(__file__).parent

BOOK1_PATH = BASE_DIR / "book1.txt"
BOOK2_PATH = BASE_DIR / "book2.txt"
BOOK3_PATH = BASE_DIR / "book3.txt"
SHAKESPEARE_PATH = BASE_DIR / "shakespeare.txt"
TEXT_PATH = BASE_DIR / "text.txt"

text_1 = BOOK1_PATH.read_text(encoding="utf-8")
text_2 = BOOK2_PATH.read_text(encoding="utf-8")
text_3 = BOOK3_PATH.read_text(encoding="utf-8")

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
main_text_3 = strip_gutenberg(text_3)

main_text = main_text_1 + "\n\n" + main_text_2 + "\n\n" + main_text_3

TEXT_PATH.write_text(main_text, encoding="utf-8")

print(len(main_text))