import os.path
from pathlib import Path
from PIL import Image


def main(path = Path('train_resize/') ):
    pure_list = []
    save_train = path
    for filename in os.listdir(save_train):
        pure_list.append(filename)

    imgs = [Image.open(os.path.join(save_train, Path(i))).convert('RGB') for i in pure_list]

    for i in range(len(imgs)):
        imgs[i] = imgs[i].resize((32, 32))
        imgs[i].save(os.path.join(save_train,pure_list[i]))


if __name__ == '__main__':
    main()