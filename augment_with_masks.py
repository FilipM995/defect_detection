
# augment data by changing the brightness of the images, and flipping them horizontally

import os
import sys
import albumentations as A
import cv2
from matplotlib import pyplot as plt
import numpy as np

def augment_images_and_masks(input_folder, output_folder, num_augmented_images):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=90, p=0.5),
        # A.RandomGamma(p=0.5),
    ])

    it=sorted(os.listdir(input_folder))

    augmented_per_image = num_augmented_images

    # def decrease_brightness(img, value=30):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)

    #     lim = 0 + value
    #     v[v < lim] = 0
    #     v[v >= lim] -= value

    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img

    for i in range(0,len(it),2):
        img= cv2.imread(os.path.join(input_folder, it[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.bitwise_not(img)
        # img = cv2.normalize(img, img,alpha=0,beta=1, norm_type=cv2.NORM_MINMAX)
        # normalized_img = cv2.imread('kolektorsdd2/train/10000.png', cv2.COLOR_BGR2RGB)
        # img = cv2.normalize(img, None, alpha=0, beta=60, norm_type=cv2.NORM_MINMAX)
        # img= decrease_brightness(img, value=90)
        # print(img)
        mask= cv2.imread(os.path.join(input_folder, it[i+1]))
        # print(mask)

        for j in range(augmented_per_image):
            transformed = transform(image=img, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']
            image_name = f"{it[i].replace('.png', f'_aug{j}.png')}"
            mask_name = f"{it[i+1].replace('GT.png', f'aug{j}_GT.png')}"
            cv2.imwrite(os.path.join(output_folder, image_name), transformed_image)
            cv2.imwrite(os.path.join(output_folder, mask_name), transformed_mask)

if __name__ == "__main__":
    input_folder = "annotations/images_and_masks"
    output_folder = "kolektorsdd2/train_augmented"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # remove all files in output folder
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))

    it=sorted(os.listdir(input_folder))
    for i in range(0,len(it),2):
        img = cv2.imread(os.path.join(input_folder, it[i]))
        mask = cv2.imread(os.path.join(input_folder, it[i+1]))

        if np.count_nonzero(mask) == 0:
            img_copy = img.copy()
            mask_copy = mask.copy()
            copy_img_name = f"{it[i].replace('.png', f'_copy.png')}"
            copy_mask_name = f"{it[i+1].replace('GT.png', f'copy_GT.png')}"
            cv2.imwrite(os.path.join(input_folder, copy_img_name), img_copy)
            cv2.imwrite(os.path.join(input_folder, copy_mask_name), mask_copy)

    num_augmented_images = 5
    augment_images_and_masks(input_folder, output_folder, num_augmented_images)