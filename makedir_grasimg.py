import os
import shutil
import pandas as pd

output_dir = "GRAS_DS"
os.makedirs(output_dir, exist_ok=True)

for ds in ["aiface", "fairface"]:
    df = pd.read_csv(f"selected_images_{ds}.csv")
    for filepath in df['file']: 
        if os.path.isfile(filepath):  
            dest_path = os.path.join(output_dir, filepath)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy(filepath, dest_path)
        else:
            print(f"File not found: {filepath}")
print("GRAS Image dataset created.")