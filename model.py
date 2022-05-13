import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier



import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('preparedData.csv')
# print(df.head())
df.drop('Unnamed: 0', axis=1, inplace=True)

x = df.drop('hospital_death', axis=1)
y = df['hospital_death']
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=10)

mlp = MLPClassifier(max_iter=500)
mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)


import pickle

pickle.dump(mlp, open('model_new.pkl', 'wb'))
model = pickle.load(open('model_new.pkl', 'rb'))
y_pred = mlp.predict(x_test)
print(y_pred)