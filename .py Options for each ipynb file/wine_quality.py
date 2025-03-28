import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv('winequality-red.csv', sep=';')

data['good_wine'] = np.where(data['quality'] >= 7, 1, 0)
data = data.drop('quality', axis=1)

X = data.drop('good_wine', axis=1)
y = data['good_wine']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)