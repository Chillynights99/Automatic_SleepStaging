import pandas as pd
import pickle
import features
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

# Load the training features
N1_train = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N1_features")
N2_train = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N2_features")
N3_train = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_N3_features")
W_train = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_W_features")
R_train = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Train_R_features")

# Load testsets
N1_test = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_N1_features")
N2_test = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_N2_features")
N3_test = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_N3_features")
W_test = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_W_features")
R_test = pd.read_pickle("C:/Users/User/Desktop/EWHADATASETS/Features/Test_R_features")

#import model
loaded_model = pickle.load(open("C:/Users/User/Desktop/code/model/1220_svmodel.sav", 'rb'))

# Normalization with StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Train에서 fit 한 후 평균값이랑 표준편차 가져와서 테스트값에 적용
sc.fit(N1_train.iloc[:, :-2])
N1_scaled = pd.DataFrame(sc.transform(N1_test.iloc[:, :-2]))
N1_scaled['label'] = "N1"

sc.fit(N2_train.iloc[:, :-2])
N2_scaled = pd.DataFrame(sc.transform(N2_test.iloc[:, :-2]))
N2_scaled['label'] = "N2"

sc.fit(N3_train.iloc[:, :-2])
N3_scaled = pd.DataFrame(sc.transform(N3_test.iloc[:, :-2]))
N3_scaled['label'] = "N3"

sc.fit(W_train.iloc[:, :-2])
W_scaled = pd.DataFrame(sc.transform(W_test.iloc[:, :-2]))
W_scaled['label'] = "W"

sc.fit(R_train.iloc[:, :-2])
R_scaled = pd.DataFrame(sc.transform(R_test.iloc[:, :-2]))
R_scaled['label'] = "R"

# Test
testsets = pd.concat([N1_scaled, N2_scaled, N3_scaled, R_scaled, W_scaled], ignore_index=True)

X_test = testsets.iloc[:, :-1]
y_test = testsets.iloc[:, -1]

pred = loaded_model.predict(X_test)

print(classification_report(y_test,pred))

f,ax = plt.subplots(figsize=(8, 7))
sns.heatmap(confusion_matrix(y_test,pred), cmap="Blues", annot=True, linewidths=.1, fmt= '.0f',ax=ax)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.show()
