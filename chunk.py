from pathlib import Path
material="78"
directory_path = Path(f"/home/ehavugim/Downloads/{material}")
from math import ceil
import pandas as pd
save_dir = Path(f"./{material}")

for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.name.endswith("csv"):
            print(f"xxxContent of {file_path.name}:\n---")
            content = file_path.read_text()
            print(f"Content of {file_path.name}:\n --")
            data=pd.read_csv(file_path,header=None)
            print(data.shape)
            maxLen=data.shape[0]
            baseFileName=file_path.name[:-4]
            chuncksize=25
            for i in range(ceil(maxLen/chuncksize)):
                datai=data.iloc[i*chuncksize:min((i+1)*chuncksize,maxLen),:]
                print(datai.shape, baseFileName,i)
                # print(save_dir)
                datai.to_csv(f'{material}/{baseFileName}_{i}.csv', index=False,header=False)
