import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "./data/rocov2"

df_train = pd.read_csv(os.path.join(DATA_PATH, "processed", "train.csv"))
df_valid = pd.read_csv(os.path.join(DATA_PATH, "processed", "valid.csv"))
df_test = pd.read_csv(os.path.join(DATA_PATH, "processed", "test.csv"))

df_cui = pd.read_csv(os.path.join(DATA_PATH, "unprocessed", "cui_mapping.csv"))

cui_freq = {}
for i in tqdm(range(len(df_train))):
    cuis = df_train.iloc[i]["CUIs"].split(";")
    for cui in cuis:
        if cui in cui_freq.keys():
            cui_freq[cui] += 1
        else:
            cui_freq[cui] = 1

k = 20
sorted_cui_freq = dict(sorted(cui_freq.items(), key=lambda item: item[1], reverse=True))
top_k_cui = dict(list(sorted_cui_freq.items())[:k])

print(top_k_cui)

top_k_cui_set = set(top_k_cui.keys())
rows_to_remove = []

for i in tqdm(range(len(df_train))):
    cuis = df_train.iloc[i]["CUIs"].split(";")
    if len(top_k_cui_set.intersection(set(cuis))) != len(cuis):
        rows_to_remove.append(i)

df_train = df_train.drop(rows_to_remove)
df_train = df_train.reset_index(drop=True)

rows_to_remove = []

for i in tqdm(range(len(df_valid))):
    cuis = df_valid.iloc[i]["CUIs"].split(";")
    if len(top_k_cui_set.intersection(set(cuis))) != len(cuis):
        rows_to_remove.append(i)

df_valid = df_valid.drop(rows_to_remove)
df_valid = df_valid.reset_index(drop=True)

rows_to_remove = []

for i in tqdm(range(len(df_test))):
    cuis = df_test.iloc[i]["CUIs"].split(";")
    if len(top_k_cui_set.intersection(set(cuis))) != len(cuis):
        rows_to_remove.append(i)

df_test = df_test.drop(rows_to_remove)
df_test = df_test.reset_index(drop=True)

df_train.to_csv(os.path.join(DATA_PATH, "processed", f"train_top_{k}_cui.csv"), index=False)
df_valid.to_csv(os.path.join(DATA_PATH, "processed", f"valid_top_{k}_cui.csv"), index=False)
df_test.to_csv(os.path.join(DATA_PATH, "processed", f"test_top_{k}_cui.csv"), index=False)

print(df_train.shape, df_valid.shape, df_test.shape)