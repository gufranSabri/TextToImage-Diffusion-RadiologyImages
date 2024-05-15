import os
import pandas as pd

phase = 'test'
c = "4"

extra = "_knee" if c != "0" else ""
df = pd.read_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cf{extra}.csv"))

print(df.columns)

knee_path = f"/Users/gufran/Developer/Projects/AI/RadiologyTextToImage/data/archive/{phase}/{c}"

knee_files = os.listdir(knee_path)

captions ={
    "0": "X-ray of normal knee",
    "1": "X-ray of knee with osteoarthritis with level 1",
    "2": "X-ray of knee with osteoarthritis with level 2",
    "3": "X-ray of knee with osteoarthritis with level 3",
    "4": "X-ray of knee with osteoarthritis with level 4",
}

for file in knee_files:
    condition = "Normal" if c == 0 else f"Osteoarthritis level {c}"
    row = {"ID": file.split(".")[0], "Caption": captions[c], "CUIs":"C1306645", "CUI_caption":"Plain x-ray", "keywords": f"X-ray, Knee, Front, {condition}"}
    df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)


df.to_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cf_knee.csv"), index=False)
df = pd.read_csv(os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cf_knee.csv"))
print(df.tail())