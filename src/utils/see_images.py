import pandas as pd
import cv2
import os

path = "./data/rocov2/processed/train_top_20_key_cf.csv"

df = pd.read_csv(path)

for i, row in df.iterrows():
    print(row["Caption"], row["CUIs"], row["keywords"])
    image_path = os.path.join('./data/rocov2/train_images', row['ID'] + (".jpg" if "ROCO" in row['ID'] else ".png"))
    img = cv2.imread(image_path)

    #apply histogram equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imshow('image', img)
    if ord("q") == cv2.waitKey(0): exit()
    cv2.destroyAllWindows()