import pandas as pd
import os

phase = 'train'
file_path = os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cf.csv")



df = pd.read_csv(file_path)

cuis_keep = ["C0040405","C1306645"]

for i, row in df.iterrows():
    cuis = row["CUIs"].split(";")
    if not any(cui in cuis_keep for cui in cuis):
        df.drop(i, inplace=True)

df.to_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cf_xray.csv"), index=False)