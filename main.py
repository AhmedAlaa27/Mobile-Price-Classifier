import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

data = pd.read_csv('Requirements/mobile_prices.csv')

print(data.head())

x = data.drop(['price_range'], axis=1)
y = data['price_range']

scalar = StandardScaler()
x = scalar.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = LogisticRegression()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)

accuracy = accuracy_score(y_test, y_predict) * 100
print(f'Accuracy = {accuracy}%')

print(classification_report(y_test, y_predict))

# print(y_predict)
