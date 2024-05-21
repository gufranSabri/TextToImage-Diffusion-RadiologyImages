
import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def calculate_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = -np.sum(entropy[entropy > 0] / gray.size * np.log2(entropy[entropy > 0] / gray.size))

    return entropy

def filter_images(images, quality_threshold, brightness_threshold):
    drop_rows = []
    for i, img in tqdm(enumerate(images)):
        img = cv2.imread(img)
        quality = calculate_image_quality(img)
        
        if quality > quality_threshold:
            brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if brightness_threshold[0] <= brightness <= brightness_threshold[1]:
                pass
            else:
                drop_rows.append(i)

    return drop_rows

def main(phase):
    df = pd.read_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top10_kc.csv"))

    images = []
    quality_threshold = 4.0  # Adjust based on desired quality
    brightness_threshold = (50, 200)  # Adjust based on desired brightness range
    for filename in df.ID:
        images.append(os.path.join("./data/rocov2", f"{phase}_images", filename+".jpg"))

    drop_rows = filter_images(images, quality_threshold, brightness_threshold)
    df.drop(drop_rows, inplace=True)

    df.to_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top10_kcf.csv"), index=False)
    print(df.shape)

main("train")
main("valid")
main("test")