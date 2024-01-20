
# augment data by changing the brightness of the images, and flipping them horizontally

import os
import albumentations as A
import cv2
import numpy as np

def augment_images_and_masks(input_folder, output_folder, num_augmented_images):

    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomBrightnessContrast(p=0.5),
        # A.RandomGamma(p=0.5),
    ])

    it=sorted(os.listdir(input_folder))

    # def decrease_brightness(img, value=30):
    #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #     h, s, v = cv2.split(hsv)

    #     lim = 0 + value
    #     v[v < lim] = 0
    #     v[v >= lim] -= value

    #     final_hsv = cv2.merge((h, s, v))
    #     img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    #     return img

    for i in range(0,min(len(it),num_augmented_images*2),2):
        img= cv2.imread(os.path.join(input_folder, it[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # normalized_img = cv2.imread('kolektorsdd2/train/10000.png', cv2.COLOR_BGR2RGB)
        # img = cv2.normalize(img, None, alpha=0, beta=60, norm_type=cv2.NORM_MINMAX)
        # img= decrease_brightness(img, value=90)
        # print(img)
        mask= cv2.imread(os.path.join(input_folder, it[i+1]))
        # print(mask)

        transformed = transform(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        cv2.imwrite(os.path.join(output_folder, it[i]), transformed_image)
        cv2.imwrite(os.path.join(output_folder, it[i+1]), transformed_mask)

if __name__ == "__main__":
    input_folder = "/kaggle/working/defect_detection/images_and_masks"
    output_folder = "/kaggle/working/defect_detection/KSDD2/train_augmented"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # remove all files in output folder
    for file in os.listdir(output_folder):
        os.remove(os.path.join(output_folder, file))
    num_augmented_images = 50
    augment_images_and_masks(input_folder, output_folder, num_augmented_images)
