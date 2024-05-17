import pandas as pd
import cv2
import os

path = "./data/rocov2/processed/train_top_20_key_cf.csv"

df = pd.read_csv(path)

for i, row in df.iterrows():
    image_path = os.path.join('./data/rocov2/train_images', row['ID'] + (".jpg" if "ROCO" in row['ID'] else ".png"))
    img = cv2.imread(image_path)
    
    print(i)
    if i < 10000: continue
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()