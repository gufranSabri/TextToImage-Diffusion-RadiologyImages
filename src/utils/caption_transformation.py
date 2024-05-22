import ollama
import pandas as pd
import os
from tqdm import tqdm

def filtered_caption(caption):
    prompt = caption + "\n\n ======================================"

    prompt += "Above is a medical caption. Do the following: \n"
    prompt += "Rewrite the caption as follows: <Image type>, <Body part>, <View of Image>, <Patient's Condition>.\n"
    prompt += "If the condition is not mentioned, just say Normal in place of Patient's Condition' \n"
    prompt += "Don't use abbreviations; example, instead of CT, say computed tomography. Try to be general, For example, instead of saying radiograph, just sat X-ray. Use simple words for view, example above, side, bottom, etc. Remove all other information, example if there is information about a disease or a condition, remove it.\n"
    prompt += "If the caption does not have the information I specified (image type, body part, view), just say 'invalid' \n"
    prompt += "Do not say anything except the new caption \n"

    response = ollama.generate(model='llama3', prompt=prompt)
    return response['response']

def main(data_path, phase):
    df = pd.read_csv(os.path.join(data_path, "processed", f"{phase}_top20.csv"))

    transformed_captions = []
    for i in tqdm(range(len(df)), desc=f"Transforming Query for {phase}"):
        caption = filtered_caption(df['Caption'][i] + " " + df["CUI_caption"][i])
        transformed_captions.append(caption)

    df['keywords'] = transformed_captions
    df = df[df['keywords'] != 'invalid']
    df.to_csv(os.path.join(data_path, "processed", f"{phase}_top20_k.csv"), index=False)

if __name__ == "__main__":
    DATA_PATH = "./data/rocov2"
    main(DATA_PATH, 'train')
    main(DATA_PATH, 'valid')
    main(DATA_PATH, 'test')