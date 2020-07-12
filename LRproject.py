import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
df = pd.read_csv('Ecommerce Customers')
print(df.columns)
print(df.info())
X = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]
y= df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)
#sns.lmplot(x='Yearly Amount Spent', y='Length of Membership', data=df)
#print(lm.coef_)
pred = lm.predict(X_test)
#print(pred)
plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
print('MSA:',metrics.mean_absolute_error(y_test, pred))
print('MSE' , metrics.mean_squared_error(y_test, pred))
print('RMSE' ,np.sqrt(metrics.mean_squared_error(y_test, pred)))
plt.show()