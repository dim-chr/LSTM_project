import pandas as pd
import numpy as np

df=pd.read_csv("TSLA.csv")
print("Number of rows and columns:", df.shape)
df.head(5)