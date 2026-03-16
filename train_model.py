import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('D:/archive (6)/House_Rent_Dataset.csv')

df = df[['BHK','Size','Bathroom','City','Furnishing Status','Rent']]

le_city = LabelEncoder()
le_furnish = LabelEncoder()

df['City'] = le_city.fit_transform(df['City'])
df['Furnishing Status'] = le_furnish.fit_transform(df['Furnishing Status'])

X = df.drop("Rent",axis=1)
y = df["Rent"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestRegressor()

model.fit(X_train,y_train)

pickle.dump(model,open("../rent_model.pkl","wb"))

print("Model trained successfully")