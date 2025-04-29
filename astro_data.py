import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv('neo.csv')
data = pd.read_csv('neo_v2.csv')

'''
print(data.head())

print(data.info())

print(data.isnull().sum())
'''

#data = data[['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude','hazardous']]

#print(data.head())

x = data[['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude']]
y = data[['hazardous']]

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
print("Classification Report : \n", classification_report(y_test, y_pred))