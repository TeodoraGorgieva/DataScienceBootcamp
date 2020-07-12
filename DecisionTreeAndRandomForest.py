import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv('loan_data.csv')
loans[loans['credit.policy']==1]['fico'].hist(bins=35, color='blue', label='Credit policy=1', alpha=0.6)
loans[loans['credit.policy']==0]['fico'].hist(bins=35, color='red', label='Credit policy=0')
plt.legend()

loans[loans['not.fully.paid']==1]['fico'].hist(bins=35, color='blue', label='Not fully paid=1', alpha=0.6)
loans[loans['not.fully.paid']==0]['fico'].hist(bins=35, color='blue', label='Not fully paid=0', alpha=0.6)

plt.figure(figsize=(11,7))
#sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

#sns.jointplot(x='fico', y='int.rate', data=loans, colors='purple')

#sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy')

cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
print(final_data.info())

X=final_data.drop('not.fully.paid', axis=1)
y=final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test, predictions))
print('\n')
print(confusion_matrix(y_test, predictions))

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test, rfc_pred))
print('\n')
print(confusion_matrix(y_test, rfc_pred))
#plt.show()
