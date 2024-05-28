from super_image import A2nModel, ImageLoader
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

# best_images = os.listdir(f'./fig/best_samples')
# worst_images = os.listdir(f'./fig/worst_samples')

# for i in tqdm(range(len(worst_images))):
#     img_best = f"{i+1}.jpg"
#     img_worst = f"{i+1}.jpg"

#     if img_best.endswith('.jpg') and img_worst.endswith(".jpg"):
#         image_best = Image.open(f'./fig/best_samples/{img_best}')
#         image_worst = Image.open(f'./fig/worst_samples/{img_worst}')

#         model = A2nModel.from_pretrained('eugenesiow/a2n', scale=4)
#         inputs_best = ImageLoader.load_image(image_best)
#         inputs_worst = ImageLoader.load_image(image_worst)

#         preds_best = model(inputs_best)
#         preds_worst = model(inputs_worst)

#         ImageLoader.save_image(preds_best, f'./fig/best_4x/{img_best.replace(".jpg","")}_4x.jpg')
#         ImageLoader.save_image(preds_worst, f'./fig/worst_4x/{img_worst.replace(".jpg","")}_4x.jpg')



#display the upscaled images side by side using numpy hstack. read using opencv
best_images = os.listdir(f'./fig/best_4x')
worst_images = os.listdir(f'./fig/worst_4x')
df = pd.read_csv("./data/rocov2/processed/test_top10_kcf.csv")

for i in tqdm(range(len(best_images))):
    img_best = f"{i+1}_4x.jpg"
    img_worst = f"{i+1}_4x.jpg"
    ground = os.path.join('./data/rocov2/test_images', df.iloc[i]['ID'] + ".jpg")

    if img_best.endswith('.jpg') and img_worst.endswith(".jpg"):
        image_best = cv2.imread(f'./fig/best_4x/{img_best}')
        image_worst = cv2.imread(f'./fig/worst_4x/{img_worst}')
        image_ground = cv2.imread(ground)

        image_best = cv2.resize(image_best, (256, 256))
        image_worst = cv2.resize(image_worst, (256, 256))
        image_ground = cv2.resize(image_ground, (256, 256))

        image = np.hstack([image_best, image_worst, image_ground])
        cv2.imshow('image', image)
        k = cv2.waitKey(0)
        if ord("q") == k: exit()
        if ord("s") == k:
            print(img_best)
            cv2.imwrite(f"./fig/best_saved/{i+1}.jpg", image_best)
            cv2.imwrite(f"./fig/worst_saved/{i+1}.jpg", image_worst)
            cv2.imwrite(f"./fig/ground_saved/{df.iloc[i]['ID']}.jpg", image_ground)
            cv2.imwrite(f"./fig/combined_saved/{i+1}.jpg", image)

        cv2.destroyAllWindows()