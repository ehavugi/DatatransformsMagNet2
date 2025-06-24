import torch
import pandas as pd
import os
from functools import lru_cache

@lru_cache(5)
def load_chunked(material="78",setnumber=1,chunck=1):
    file_path=f"{material}/{material}_{setnumber}_B_{chunck}.csv"
    # print("searching", file_path)
    if os.path.exists(file_path):
        dataB=pd.read_csv(f"{material}/{material}_{setnumber}_B_{chunck}.csv",header=None)
        dataH=pd.read_csv(f"{material}/{material}_{setnumber}_H_{chunck}.csv",header=None)
        dataT=pd.read_csv(f"{material}/{material}_{setnumber}_T_{chunck}.csv",header=None)

        return {"dataB":dataB,"dataH":dataH,"dataT":dataT}
    else:
        return None
# print(load_chunked('78',1,1))


def collate_fn(batch):
    if len(batch[0])==5:
        dataB=torch.vstack([item[0] for item in batch if len(item[0])>0])
        dataH=torch.vstack([item[1] for item in batch if len(item[0])>0])
        dataB_scaler=torch.vstack([item[2] for item in batch if len(item[0])>0])
        dataT=torch.vstack([item[3] for item in batch if len(item[0])>0])
        dataH_scaler=torch.vstack([item[4] for item in batch if len(item[0])>0])


        return dataB,dataH,dataB_scaler,dataT,dataH_scaler
def pandas2tensor(x):
    x = x.to_numpy()
    py_tensor = torch.from_numpy(x)
    py_tensor=py_tensor.to(torch.float32)
    return py_tensor.view(-1,len(x))
@lru_cache(maxsize=128)
def load_all(material):
    data={}
    for i in range(1,8):
        chunck=0
        end=False
        while not end:
            if chunck==0:
                data[i] = load_chunked(material=material,setnumber=i,chunck=chunck)
                chunck+=1
            else:
                # print("material",material,"i",i,"chunck",chunck)
                loaded= load_chunked(material=material,setnumber=i,chunck=chunck)
                # print("loaded",loaded)
                if  loaded is not None:
                    # print('shapes',type(data[i]['dataB']),type(loaded['dataB']))
                    data[i]['dataB']= pd.concat([data[i]['dataB'], loaded['dataB']], ignore_index=True)
                    data[i]['dataH']= pd.concat([data[i]['dataH'], loaded['dataH']], ignore_index=True)
                    data[i]['dataT']= pd.concat([data[i]['dataT'], loaded['dataT']], ignore_index=True)


                    chunck=chunck+1
                else:
                    chunck=0
                    end=True

        # print("done with while", end, i)


    return data 


class MagNetChallange2(torch.utils.data.Dataset):
    """docstring for MagnetChallenge2"""
    def __init__(self,material,size=80,skip=10,magnetx=True):
        self.material=material
        self.maps=load_all(material)
        self.inputsize=size+1
        self.skip=skip
        self.stop_indices=1
        self.cumulative={}
        self.columns={}
        self.rows={}
        self.magnetx=magnetx
        self.saved=False
        self.reset_indices()
        self.saved_data=[0 for x in range(self.__len__())]
        # print("saved", self.saved_data)
        self.set_subset()
        print(f"data loaded for {material}")
    def group_select(self,i):
        if i>max(self.cumulative.values()):
            return None
        else:
            for j in range(8):
                if i>=self.cumulative.get(j,0) and i<=self.cumulative.get(j+1,0):
                    return j+1
    def set_subset(self):
        self.saved=False
        for i in range(self.__len__()):
            self.saved_data[i]=self.__getitem__(i)
        self.saved=True

    def row_select(self,i):
        x=self.group_select(i)
        stop_indices=self.stop_indices
        skip=self.skip
        inputsize=self.inputsize
        if x is None:
            return None
        base_x=i-self.cumulative.get(x-1,0) # number of (set points) in current group of points
        data_in_a_column=(self.columns[x]) # data in each column in this group

        # data available in each column (accounting for stop indices ratio)
        data_available_a_column=int(data_in_a_column*stop_indices)

        # points in a column accounting for input size and skip multiplier 
        # skip=0, points would be back to back
        points_in_a_column=data_available_a_column//((inputsize*(1+skip)))
        row_index=base_x//points_in_a_column

        row_index=min(row_index,self.rows[x]-1)
        column_base=base_x-row_index*points_in_a_column
        start=column_base*(inputsize*(1+skip))
        end=start+inputsize
        # if (i*100/self.__len__())%10==0:
            # print(i,(i*100/self.__len__()),"%","done" )

        return x,row_index,start,end
    def reset_indices(self):
        cumulative={}
        columns={}
        rows={}
        for i in range(1,8):
            inputsize=self.inputsize
            skip=self.skip
            stop_indices=self.stop_indices
            x=self.maps[i]['dataB'].shape[0],self.maps[i]['dataB'].shape[1]
            cumulative[i]=cumulative.get(i-1,0)+x[0]*(int(x[1]*stop_indices)//(inputsize*(1+skip)))
            columns[i]=x[1]
            rows[i]=x[0]
        self.columns=columns
        self.cumulative=cumulative
        self.rows=rows

    def load(self,loc):
        return 
    def __len__(self):
        return max(self.cumulative.values())-2
    def __getitem__(self,idx):
        if self.saved:
            return self.saved_data[idx]
        try:
            if self.magnetx:
                # would format data in format used in tutorial 1
                x,row_index,start,end=self.row_select(idx) # currently buggy for index 0, and len(x)
                dataB=self.maps[x]['dataB'].iloc[row_index,start:end-1]
                dataH=self.maps[x]['dataH'].iloc[row_index,start:end-1]
                B_scaler=self.maps[x]['dataB'].iloc[row_index,end-1:end]
                H_scaler=self.maps[x]['dataH'].iloc[row_index,end-1:end]
                dataT=self.maps[x]['dataT'].iloc[row_index]
                # with open('logs.txt',"a") as f:
                #     out=(pandas2tensor(dataB),pandas2tensor(dataH),pandas2tensor(B_scaler),pandas2tensor(dataT),pandas2tensor(H_scaler))
                #     out_str=str(idx)+","+",".join([str(xi.shape) for xi in out])
                #     f.write(out_str+"\n")
                return pandas2tensor(dataB),pandas2tensor(dataH),pandas2tensor(B_scaler),pandas2tensor(dataT),pandas2tensor(H_scaler)
            else:
                # generic data format
                x,row_index,start,end=self.row_select(idx)
                dataB=self.maps[x]['dataB'].iloc[row_index,start:end]
                dataH=self.maps[x]['dataH'].iloc[row_index,start:end]
                dataT=self.maps[x]['dataT'].iloc[row_index]
                return pandas2tensor(dataB),pandas2tensor(dataH),pandas2tensor(dataT)
        except Exception as e:
            print(e)
            with open("logs.txt",'a') as f:
                f.write(f"Error occured  {idx}\n")
            return None,None,None,None,None

if __name__=="__main__":
    data=MagNetChallange2("78")
    data[0]

