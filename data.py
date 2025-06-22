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

def pandas2tensor(x):
	x = x.to_numpy()
	pytorch_tensor = torch.from_numpy(x)
	return pytorch_tensor
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

		# print("done with while", end)


	return data 


class MagNetChallange2(torch.utils.data.Dataset):
	"""docstring for MagnetChallenge2"""
	def __init__(self,material,size=80,magnetx=True):
		self.material=material
		self.maps=load_all(material)
		self.inputsize=size+1
		self.skip=10
		self.stop_indices=1
		self.cumulative={}
		self.columns={}
		self.magnetx=magnetx
		self.reset_indices()
		
	def group_select(self,i):
	    if i>max(self.cumulative.values()):
	        return None
	    else:
	        for j in range(8):
	            if i>self.cumulative.get(j,0) and i<self.cumulative.get(j+1,0):
	                return j+1

	def row_select(self,i):
	    x=self.group_select(i)
	    stop_indices=self.stop_indices
	    skip=self.skip
	    inputsize=self.inputsize
	    if x is None:
	    	return None
	    base_x=i-self.cumulative.get(x-1,0)
	    data_in_a_column=(self.columns[x])
	    # print("data in a columns", data_in_a_column)
	    data_available_a_column=int(data_in_a_column*stop_indices)
	    # print("data_available a columns", data_available_a_column)
	    points_in_a_column=data_available_a_column//((inputsize*(1+skip)))
	    # print("points_in a columns", points_in_a_column)
	    row_index=base_x//points_in_a_column
	    # x[0]*(//(inputsize*(1+skip)))
	    column_base=base_x-row_index*points_in_a_column
	    start=column_base*(inputsize*(1+skip))
	    end=start+inputsize
	    # print(column_base, start,end)
	    return x,row_index,start,end
	def reset_indices(self):
		cumulative={}
		columns={}
		for i in range(1,8):
		    inputsize=self.inputsize
		    skip=self.skip
		    stop_indices=self.stop_indices
		    x=self.maps[i]['dataB'].shape[0],self.maps[i]['dataB'].shape[1]
		    cumulative[i]=cumulative.get(i-1,0)+x[0]*(int((x[1])*stop_indices)//(inputsize*(1+skip)))
		    columns[i]=x[1]
		    # print(i,x,x[0]*x[1]//(inputsize*(1+skip)),cumulative[i])
		self.columns=columns
		self.cumulative=cumulative

	def load(self,loc):
		return 
	def __len__(self):
		return max(self.cumulative.values())-2
	def __getitem__(self,idx):
		if self.magnetx:
			# would format data in format used in tutorial 1
			x,row_index,start,end=self.row_select(idx+1) # currently buggy for index 0, and len(x)
			dataB=self.maps[x]['dataB'].iloc[row_index,start:end-1]
			dataH=self.maps[x]['dataH'].iloc[row_index,start:end-1]
			B_scaler=self.maps[x]['dataB'].iloc[row_index,end-1:end]
			H_scaler=self.maps[x]['dataH'].iloc[row_index,end-1:end]
			dataT=self.maps[x]['dataT'].iloc[row_index]

			return pandas2tensor(dataB),pandas2tensor(dataH),pandas2tensor(B_scaler),pandas2tensor(dataT),pandas2tensor(H_scaler)
		else:
			# generic data format
			x,row_index,start,end=self.row_select(idx)
			dataB=self.maps[x]['dataB'].iloc[row_index,start:end]
			dataH=self.maps[x]['dataH'].iloc[row_index,start:end]
			dataT=self.maps[x]['dataT'].iloc[row_index]
			return pandas2tensor(dataB),pandas2tensor(dataH),pandas2tensor(dataT)

if __name__=="__main__":
    data=MagNetChallange2("78")

