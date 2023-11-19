import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

data = pd.read_csv('fooddata.csv', encoding = "ISO-8859-1")
#print(data.describe())

y = data['SOLD_QTY'] 

le = preprocessing.LabelEncoder()

myData=np.genfromtxt('fooddata.csv', delimiter=",", dtype ="|a20" ,skip_header=1);

for i in range(3):
    myData[:,i] = le.fit_transform(myData[:,i])
	
X_train, X_test, y_train, y_test = train_test_split(myData, y ,test_size=0.3, random_state=101)

#print(X_test)

rf = RandomForestRegressor(max_depth=2, random_state=101)

rf.fit(X_train,y_train)

y_pred = rf.predict(X_test)

rse = metrics.r2_score(y_test,y_pred) 

print('')
print('R2 Score : '+ str(rse))

###### For Graph Ploting

data2 = pd.read_csv('fooddata_subset.csv', encoding = "ISO-8859-1")
x2 = data2['MON']
y2 = data2['SOLD_QTY'] 

myData2=np.genfromtxt('fooddata_subset.csv', delimiter=",", dtype ="|a20" ,skip_header=1);

for i in range(3):
    myData2[:,i] = le.fit_transform(myData2[:,i])
	
X_train2, X_test2, y_train2, y_test2 = train_test_split(myData2, y2 ,test_size=0.3, random_state=101)

X_train1, X_test1, y_train1, y_test1 = train_test_split(data2, y2 ,test_size=0.3, random_state=101)

rf2 = RandomForestRegressor(max_depth=2, random_state=0) 

rf2.fit(X_train2,y_train2)

y_pred2 = rf2.predict(X_test2)

a,b = X_test1['MON'], X_test1['SOLD_QTY']

c,d = X_test1['MON'] , y_pred2

plt.plot(a,b,label='Test Data', color= 'blue')

plt.plot(c,d,label='Prediction', color= 'orange')

plt.xlabel("Month")
plt.ylabel("Sold Quantity")

plt.title('Month vs Sold Quantity')

plt.legend()

plt.show()

print(y_test2)
print(y_pred2)

rse2 = metrics.r2_score(y_test2,y_pred2) 

#print(rse2)

