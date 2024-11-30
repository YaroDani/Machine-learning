import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df= pd.read_csv("Titanic-Dataset.csv")
df.drop(["Embarked", "Cabin", "Ticket", "Name", "Parch", "Fare", "PassengerId"], axis=1, inplace=True)
df["Age"]=df["Age"].fillna(28)
#print(df.info)



def fill_fel_mel(Sex):
    if Sex=="female":
        return 0
    if Sex=="male":
        return 1

df['Sex'] = df['Sex'].apply(fill_fel_mel)
x = df.drop(['Survived'], axis=1)
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('''
''')
print(accuracy_score(y_test, y_pred) * 100)

#print(df.info())


