import pandas as pd
import os

def main(phase):
    file_path = os.path.join("./data/rocov2", "processed", f"{phase}_top10_k.csv")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    new_lines = []
    for line in lines:
        line = line.replace("\n","")
        if len(new_lines) == 0:
            new_lines.append(line)
            continue

        if not line.startswith("ROCO") and "ROCO" in line:print(line)
        if line.startswith("ROCO"):
            new_lines.append(line)
        elif len(new_lines) > 0:
            new_lines[-1] = new_lines[-1].strip() + " " + line.strip()

    print(len(new_lines))

    final_new_lines = []
    count = 0
    for i, line in enumerate(new_lines):
        if i == 0:
            final_new_lines.append(line)
            
        elif "computed" in line.lower() or "tomography" in line.lower() or "x-ray" in line.lower() or "radiograph" in line.lower() or "xray" in line.lower():
            count += 1
            final_new_lines.append(line)
        # else:
        #     print(line)

    print(count)
    print()

    with open(os.path.join("./data/rocov2", "processed", f"{phase}_top10_kc.csv")
    , 'w') as f:
        for i, line in enumerate(final_new_lines):
            if not line[-1] == '"' and i != 0:line += '"'
                
            f.write(line+"\n")

main("train")
main("valid")
main("test")