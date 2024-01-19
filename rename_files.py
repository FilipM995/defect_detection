import os
import re
import argparse

args = argparse.ArgumentParser()
args.add_argument('--images_masks', type=str)


is_images = args.parse_args().images_masks
if is_images == "masks":
    images_path = '/home/filip/defect-detection/source/mixed_supervision/annotations/masks'
elif is_images == "images":
    images_path = '/home/filip/defect-detection/source/mixed_supervision/annotations/images'



for i, img in enumerate(os.listdir(images_path)):
    # rename image

    if is_images == "images":
        img_number = int(re.findall(r'\d+', img)[1])
        img_name = f'{img_number}.png'
    elif is_images == "masks":
        img_number = int(re.findall(r'\d+', img)[0])
        img_name = f'{img_number}_GT.png'

    os.rename(os.path.join(images_path, img), os.path.join(images_path, img_name))