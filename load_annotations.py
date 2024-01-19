import os
import re
from PIL import Image


images_path = '/home/filip/defect-detection/source/mixed_supervision/annotations/images'
output_path = '/home/filip/defect-detection/source/mixed_supervision/annotations/images_and_masks'

mask_path = '/home/filip/defect-detection/source/mixed_supervision/annotations/masks'

pattern = re.compile(r'\d+') # find all numbers in string

img_numbers = []

for i, img in enumerate(os.listdir(images_path)):
    # find number in image name
    match = pattern.search(img)
    if match:
        img_numbers.append(int(match.group()))

    # resize image


    resized_img = Image.open(os.path.join(images_path, img))
    resized_img = resized_img.resize((229,645))
    resized_img.save(os.path.join(output_path, img))


mask_numbers = []

for i, mask in enumerate(os.listdir(mask_path)):
    # find number in mask name
    match = pattern.search(mask)
    if match:
        mask_numbers.append(int(match.group()))

    # resize mask
    resized_mask = Image.open(os.path.join(mask_path, mask))
    resized_mask = resized_mask.resize((229,645))
    resized_mask.save(os.path.join(output_path, mask))


for i in sorted(img_numbers):
    if i not in mask_numbers:
        # create empty 0 1 mask
        mask = Image.new('1', (229,645))
        mask_name = f'{i}_GT.png'
        mask.save(os.path.join(output_path, mask_name))