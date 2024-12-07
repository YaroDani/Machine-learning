import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("Housing.csv")

df_dummies=pd.get_dummies(df,columns=None, drop_first=True)
#print(df_dummies.info())

x = df_dummies.drop(["price"], axis=1)
y = df_dummies["price"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, train_size=0.3)

model = RandomForestClassifier(n_estimators=125, max_depth=12)  # n_estimators: кількість дерев, max_depth: глибина
model.fit(x_test,y_test)
y_pred=model.predict(x_test)
acc_score=accuracy_score(y_test, y_pred) * 100

categories = ["Діаграма точності"]
values = [acc_score]
plt.bar(categories, values)
plt.ylabel("Точність")
plt.show()