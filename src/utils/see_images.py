import pandas as pd
import cv2
import os

path = "./data/rocov2/processed/test_top10_kcf.csv"

df = pd.read_csv(path)

for i, row in df.iterrows():
    print(i+1)
    image_path = os.path.join('./data/rocov2/test_images', row['ID'] + (".jpg" if "ROCO" in row['ID'] else ".png"))
    img = cv2.imread(image_path)


    #apply histogram equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    #resize to 256x256
    img = cv2.resize(img, (256, 256))
    
    cv2.imshow('image', img)
    if ord("q") == cv2.waitKey(0): exit()
    cv2.destroyAllWindows()