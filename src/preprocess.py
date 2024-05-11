import os
import pandas as pd
from tqdm import tqdm

def preprocess(data_path, phase='train'):
    df_cui_captions = pd.read_csv(os.path.join(data_path, "unprocessed", f"cui_mapping.csv"))
    df_captions = pd.read_csv(os.path.join(data_path, "unprocessed", f"{phase}_captions.csv"))
    df_concepts = pd.read_csv(os.path.join(data_path, "unprocessed", f"{phase}_concepts.csv"))
    df = pd.merge(df_captions, df_concepts, on='ID')

    cui_captions = []
    rows_to_drop = []
    for i in tqdm(range(len(df)), desc=f"Getting CUI captions for {phase}"):
        cuis = df.iloc[i]["CUIs"].split(";")
        captions = []
        for cui in cuis:
            if len(df_cui_captions[df_cui_captions['CUI'] == cui]['Canonical name'].values) == 0:
                if len(cuis) == 1: rows_to_drop.append(i)
                df.at[i, "CUIs"] = str(df.at[i, "CUIs"]).replace(cui, "").replace(";;", ";").strip(";")
                continue

            captions.append(df_cui_captions[df_cui_captions['CUI'] == cui]['Canonical name'].values[0])
        
        if len(captions)!=0: cui_captions.append(";".join(captions))

    df = df.drop(rows_to_drop)
    df['CUI_caption'] = cui_captions

    df.to_csv(os.path.join(data_path, f"{phase}.csv"), index=False)

if __name__ == "__main__":
    DATA_PATH = "./data/rocov2"
    preprocess(DATA_PATH, 'train')
    preprocess(DATA_PATH, 'valid')
    preprocess(DATA_PATH, 'test')