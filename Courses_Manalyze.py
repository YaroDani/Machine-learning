
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from Titanic_survival import x_train, x_test, y_train, y_test, y_pred

df = pd.read_csv("StudentsPerformance.csv")
df.drop(["race/ethnicity","lunch"], axis=1, inplace=True )
#print(df.info())
df_dummies=pd.get_dummies(df,columns=None, drop_first=True)
print(df_dummies.info())

x = df_dummies.drop(["test preparation course_none"], axis=1)
y = df_dummies["test preparation course_none"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.35, train_size=0.3)
model=RandomForestClassifier()
model.fit(x_test,y_test)
y_pred=model.predict(x_test)
print(accuracy_score(y_test, y_pred) * 100)