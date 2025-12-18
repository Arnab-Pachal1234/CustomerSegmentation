

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('new.csv')
df.head()
#the shape of the dataset
print("The shape of the dataset is:", df.shape)
#checking for null values
print("Null values in each column:\n", df.isnull().sum())
#summary statistics of the dataset
print(df.describe())
#checking for null values in each column
for col in df.columns:
    temp = df[col].isnull().sum()
    if temp > 0:
        print(f'Column {col} contains {temp} null values.')
#dropping rows with null values
df.dropna(inplace=True)
print("Shape of the dataset after dropping null values:", df.shape)

#find total Number of unique values in each column
for col in df.columns:
    unique_values = df[col].nunique()
    print(f'Column {col} has {unique_values} unique values.')

parts = df["Dt_Customer"].str.split("-", n=3, expand=True)
df["day"] = parts[0].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[2].astype('int')

#drop unnecessary columns
df.drop(['Z_CostContact', 'Z_Revenue', 'Dt_Customer'],
        axis=1,
        inplace=True)