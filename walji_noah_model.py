import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load CSV File
df = pd.read_csv('heart_2020_cleaned.csv')

# Pre Prossessing

# First check for null values
if (df.isnull().values.any()):
    print("Null Value Detected Potential Value Missing")

else:
    print("No Null/Missing Values")

# Get Data Info
print()
print("Inital Info Of Dataset:")
df.info()
print()

# Turn Category Data To Finite Data
label = LabelEncoder()
# Splits the Variables into seperate features based on category
df=pd.get_dummies(data=df,columns=["AgeCategory","Diabetic","GenHealth","Race"])
# Sets the Yes/No Variables to 0 and 1
df_col = df.select_dtypes('object')
df[df_col.columns] = df[df_col.columns].apply(label.fit_transform)

# Removes Features Not Selected/ Not Important to The Problem
df = df.drop(["AlcoholDrinking","GenHealth_Excellent","GenHealth_Very good","Race_Asian","Race_Other","Race_Hispanic","Race_Black","Diabetic_Yes (during pregnancy)"], axis=1)

print("Final Info Of Dataset")
df.info()
print()

# Set Training and Testing Data
# 70/30
X_train, X_test, y_train, y_test = train_test_split(df.drop(['HeartDisease'], axis=1), df['HeartDisease'], test_size = 0.3, random_state = 101)

#Scale Features
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Logistic Regression Model Model Score Of 91% Which Is Good
model = LogisticRegression()
model.fit(X_train, y_train)
rsq = model.score(X = X_test, y= y_test)
rsqTrain = model.score(X=X_train,y=y_train)
print("Logistic Regression Score (Test Data): " + str(rsq))
print("Logistic Regression Score (Train Data): " + str(rsqTrain))
print("Model Coefficent: " + str(model.coef_))
print("Model Intercept: " + str(model.intercept_))
fitted_values= model.predict(X=X_test)
plt.scatter(fitted_values, y_test - fitted_values)
plt.show()

print()

# DecisionTree (Entropy) Model Score of 86% Which is not good
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)
rsq = model.score(X = X_test, y= y_test)
rsqTrain = model.score(X=X_train,y=y_train)
print("Decision Tree Score (Test Data): " + str(rsq))
print("Decision Tree Score (Train Data): " + str(rsqTrain))
fitted_values= model.predict(X=X_test)
plt.figure(figsize=(12, 6))
plot_tree(model, max_depth=5)
plt.show()

