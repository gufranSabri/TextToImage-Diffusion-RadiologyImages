import pandas as pd
import cv2
import os


path1 = "./data/rocov2/processed/train_top10_kcf.csv"
path2 = "./data/rocov2/processed/test_top10_kcf.csv"
path3 = "./data/rocov2/processed/valid_top10_kcf.csv"

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)

images_to_keep = df1["ID"].tolist() + df2["ID"].tolist() + df3["ID"].tolist()

image_folder = "./data/rocov2/train_images"

for img in os.listdir(image_folder):
    if img.split(".")[0] not in images_to_keep:
        os.remove(os.path.join(image_folder, img))

image_folder = "./data/rocov2/test_images"

for img in os.listdir(image_folder):
    if img.split(".")[0] not in images_to_keep:
        os.remove(os.path.join(image_folder, img))

image_folder = "./data/rocov2/valid_images"

for img in os.listdir(image_folder):
    if img.split(".")[0] not in images_to_keep:
        os.remove(os.path.join(image_folder, img))