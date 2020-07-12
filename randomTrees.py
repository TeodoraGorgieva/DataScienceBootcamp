import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('kyphosis.csv')
sns.pairplot(df, hue='Kyphosis')

X=df.drop('Kyphosis', axis=1)
y=df['Kyphosis']
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.3)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
#print(confusion_matrix(y_test, predictions))
#print('\n')
#print(classification_report(y_test, predictions))

rfc=RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfcPredictions = rfc.predict(X_test)
print(confusion_matrix(y_test, rfcPredictions))
print('\n')
print(classification_report(y_test, rfcPredictions))

