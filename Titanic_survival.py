import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df= pd.read_csv("Titanic-Dataset.csv")
df.drop(["Embarked", "Cabin", "Ticket", "Name", "Parch", "Fare"], axis=1, inplace=True)
df["Age"].fillna(28.0, inplace=True)
#print(df.info)

x = df.drop(['Survived'], axis=1)
y = df['Survived']

def fill_fel_mel(Sex):
    if Sex=="female":
        return 0
    if Sex=="male":
        return 1
df['Sex'] = df['Sex'].apply(fill_fel_mel)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

sc = StandardScaler()
x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_test, x_train)



