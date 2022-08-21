import argparse
import glob
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

# TRATINING SETS always append on list first and then transform into pandas dataframe, it is costly to accumulate in dataframe!!
N1_raw = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N1_features")
N2_raw = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N2_features")
N3_raw = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N3_features")
W_raw = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_W_features")
R_raw = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_R_features")

pd.set_option('precision', 8)       # 원래대로 하면 6 digits까지밖에 안나옴
scaled_N1 = pd.DataFrame(scaler.fit_transform(N1_raw.iloc[:, :-2]))
scaled_N2 = pd.DataFrame(scaler.fit_transform(N2_raw.iloc[:, :-2]))
scaled_N3 = pd.DataFrame(scaler.fit_transform(N3_raw.iloc[:, :-2]))
scaled_W = pd.DataFrame(scaler.fit_transform(W_raw.iloc[:, :-2]))
scaled_R = pd.DataFrame(scaler.fit_transform(R_raw.iloc[:, :-2]))

# 다시 레이블 붙여주기
scaled_N1['label'] = "N1"
scaled_N2['label'] = "N2"
scaled_N3['label'] = "N3"
scaled_W['label'] = "W"
scaled_R['label'] = "R"

dataset_df = pd.concat([scaled_N1, scaled_N2, scaled_N3, scaled_W, scaled_R], ignore_index=True)

X = dataset_df.iloc[:, :-1]
y = dataset_df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# Best parameter : C=100 gamma=10
model = SVC(kernel = 'rbf', C = 100, gamma = 10)
model.fit(X_train, y_train)

# Save the model
filename = "C:/Users/User/Desktop/code/model/1220_svmodel.sav"
pickle.dump(model, open(filename, 'wb'))

pred = model.predict(X_test)

f,ax = plt.subplots(figsize=(8, 7))
sns.heatmap(confusion_matrix(y_test,pred), cmap="Blues", annot=True, linewidths=.1, fmt= '.0f',ax=ax)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()

print(classification_report(y_test,pred))
