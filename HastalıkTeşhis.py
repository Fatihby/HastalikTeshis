import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("hasta_verileri.csv")
df['Cinsiyet'] = df['Cinsiyet'].map({'Erkek': 0, 'Kadın': 1})
df['Şeker'] = df['Şeker'].map({'Normal': 0, 'Yüksek': 1})
df['Sigara'] = df['Sigara'].map({'Hayır': 0, 'Evet': 1})

y = df['Teşhis'].values
X = df.drop(['Teşhis', 'Ad'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(df.describe())
print(df['Teşhis'].value_counts())
teşhisler = df['Teşhis'].unique()
teşhis_sayısı = df['Teşhis'].value_counts()

plt.bar(teşhisler, teşhis_sayısı)
plt.title('Teşhis Dağılımı')
plt.xlabel('Teşhis')
plt.ylabel('Sayı')
plt.show()

df.groupby('Teşhis')['Şeker'].mean().plot(kind='bar')
plt.title('Teşhis ve Şeker Seviyesi İlişkisi')
plt.xlabel('Teşhis')
plt.ylabel('Ortalama Şeker Seviyesi')
plt.show()

df.groupby('Teşhis')['Sigara'].mean().plot(kind='bar')
plt.title('Teşhis ve Sigara İçme Durumu İlişkisi')
plt.xlabel('Teşhis')
plt.ylabel('Sigara İçme Oranı')
plt.show()

# Decision Tree Classifier
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Model Doğruluğu:", accuracy_dt)

# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Model Doğruluğu:", accuracy_lr)

# Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Model Doğruluğu:", accuracy_rf)

# Support Vector Classifier
model_svc = SVC()
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)
accuracy_svc = accuracy_score(y_test, y_pred_svc)
print("Support Vector Model Doğruluğu:", accuracy_svc)

# K-Nearest Neighbors (KNN) Classifier
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("K-Nearest Neighbors Model Doğruluğu:", accuracy_knn)