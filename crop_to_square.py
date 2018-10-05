import os
from PIL import Image


def _crop(file_path):
    img = Image.open(file_path)
    width, height = img.size
    if width == height:
        img.close()
        return
    length = min(width, height)

    left = (width - length) // 2
    upper = (height - length) // 2
    right = left + length
    lower = upper + length

    box = (left, upper, right, lower)
    cropped = img.crop(box)
    img.close()
    cropped.save(file_path)


def crop_all(dir):
    for root, _, files in os.walk(dir):
        for f in files:
            extension = f.lower()[-4:]
            if extension not in [".jpg", ".png"]:
                continue
            _crop(os.path.join(root, f))
