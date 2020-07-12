import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
ad_data = pd.read_csv("advertising.csv")
#print(ad_data.head())
ad_data['Age'].hist()
#sns.jointplot(x='Daily Time Spent on Site', y='Age', data=ad_data, color='red', kind='kde')
#sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green')
#sns.pairplot(data=ad_data, hue='Clicked on Ad')
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)
lm = LogisticRegression()
lm.fit(X_train, y_train)
pred = lm.predict(X_test)
print(classification_report(y_test,pred))
plt.show()