import pandas as pd
import os

phase = 'valid'
file_path = os.path.join("./data/rocov2", "processed", f"{phase}_top_20_cui_keywords.csv")

with open(file_path, 'r') as f:
    lines = f.readlines()
    

new_lines = []
for line in lines:
    line = line.replace("\n","")
    if len(new_lines) == 0:
        new_lines.append(line)
        continue

    if not line.startswith("ROCO") and "ROCO" in line:print(line)
    if line.startswith("ROCO") and ("invalid" not in line or "Invalid" not in line):
        new_lines.append(line)
    elif not "Note" in line and len(new_lines) > 0:
        if "ROCO" in line:print(line.startswith("ROCO"))
        new_lines[-1] = new_lines[-1].strip() + " " + line.strip()

final_new_lines = []
count = 0
for i, line in enumerate(new_lines):
    if i == 0: 
        final_new_lines.append(line)
        
    elif "Radiograph" in line or "Computed" in line or "Tomography" in line or "X-ray" in line or "radiograph" in line or "computed" in line or "tomography" in line or "x-ray" in line:
        count += 1
        final_new_lines.append(line)

print(count)

with open(os.path.join("./data/rocov2", "processed", f"{phase}_top_20_key_cleaned.csv")
, 'w') as f:
    for i, line in enumerate(final_new_lines):
        if not line[-1] == '"' and i != 0:line += '"'
            
        f.write(line+"\n")
