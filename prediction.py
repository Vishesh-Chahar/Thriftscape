#excel to py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#obj to load book
a = pd.read_csv(r'E:\Folders\thapar waala maal\sem 5\Machine Learning\project files\7004_1.csv', lineterminator='\n')
#a is the dataframe
scale = MinMaxScaler()

ar = a.columns.get_loc("cost")
ar2 = a.columns.get_loc("costb")
Y = a.iloc[:,ar:ar2]
Y_new = scale.fit_transform(Y)
X = a.iloc[:,a.columns.get_loc("brand_id"):a.columns.get_loc("brand_id")+1]

a['scaled']=Y_new

X_train,X_test,Y_train,Y_test = train_test_split(X, Y_new, test_size = 0.25,train_size = 0.75, random_state = True, shuffle = True)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

read = int(input('Brand_id:\n'))

Y_pred = lr.predict(X_test)

plt.scatter(X_train, Y_train, color='c') 
plt.plot(X_test, Y_pred, color='b') 

#method 2
X1 = X_test.values
Y1 = Y_test
z = np.poly1d(np.polyfit(X1.flatten(),Y1.flatten(),2))
z2 = np.poly1d(np.polyfit(X1.flatten(),Y1.flatten(),3))
z3 = np.poly1d(np.polyfit(X1.flatten(),Y1.flatten(),4))

xp = np.linspace(0, 40, 1000)
_ = plt.plot(X1, Y1, '.', xp, z(xp), '-', xp, z2(xp), '--', xp, z3(xp), 2)
plt.ylim(0,1)


plt.scatter(X_train, Y_train,color='c') 
plt.show()

#print(a['scaled'])
print(a.loc[round(a['scaled'],2)==round(z2(read),2),'cost'])


