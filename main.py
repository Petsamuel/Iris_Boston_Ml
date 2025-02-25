import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
df.head()

#create label 
y = df['class']
X = df.drop('class', 'preg',  axis=1)
feature = df['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi']

X.head()