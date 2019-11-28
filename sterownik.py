import kck_cards
import pure_reader
from pathlib import Path

def main():
    path = Path('pure_rotated_resized/')
    kck_cards.main()
    pure_reader.main()
    pure_reader.main(path)

if __name__ == '__main__':
    main()