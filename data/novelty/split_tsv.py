import pandas as pd
import numpy as np

df=pd.read_csv("novel.tsv",sep="\t")
shuffled_df=df.sample(frac=1.0,random_state=42)
dfs=np.array_split(shuffled_df,10)
for i,sub_df in enumerate(dfs):
	sub_df.to_csv(f'test{i}.tsv',index=False,sep="\t")
