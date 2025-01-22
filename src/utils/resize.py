import os
from PIL import Image
import cv2
import numpy as np

def make_image_square(image):
    width, height = image.size
    if width == height:
        return image

    if width > height:
        new_image = Image.new("RGB", (width, width), (0, 0, 0))
        new_image.paste(image, (0, (width - height) // 2))
    else:
        new_image = Image.new("RGB", (height, height), (0, 0, 0))
        new_image.paste(image, ((height - width) // 2, 0))

    # #display image using opencv
    # #convert to numpy first
    # new_image = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)

    # cv2.imshow("image", new_image)
    # cv2.waitKey(0)


    return new_image

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                squared_img = make_image_square(img)
                squared_img.save(file_path)  # Overwrite the original image

# Example usage:
folder_path = '/Users/gufran/Developer/Projects/AI/RadiologyTextToImage/data/rocov2/train_images'
process_images_in_folder(folder_path)