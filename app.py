import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.title("From Votes to Value: Restaurant Analytics")

# Load Dataset
df = pd.read_csv("restaurant_analytics.csv")

st.header("Dataset Preview")
st.write(df.head())

# Data Cleaning
df["rate"] = df["rate"].str.replace("/5","")
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

st.header("Dataset Summary")
st.write(df.describe())

# ----------- EDA Section -----------

st.header("Exploratory Data Analysis")

# Restaurant type distribution
st.subheader("Restaurant Type Distribution")
fig, ax = plt.subplots()
sns.countplot(x="listed_in(type)", data=df)
plt.xticks(rotation=45)
st.pyplot(fig)

# Ratings distribution
st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
sns.histplot(df["rate"], bins=10)
st.pyplot(fig)

# Votes by restaurant type
st.subheader("Votes by Restaurant Type")
fig, ax = plt.subplots()
sns.barplot(x="listed_in(type)", y="votes", data=df)
plt.xticks(rotation=45)
st.pyplot(fig)

# Cost vs rating
st.subheader("Cost vs Rating")
fig, ax = plt.subplots()
sns.scatterplot(
    x="approx_cost(for two people)",
    y="rate",
    hue="book_table",
    data=df
)
st.pyplot(fig)

# Online vs Offline ratings
st.subheader("Online Order vs Rating")
fig, ax = plt.subplots()
sns.boxplot(x="online_order", y="rate", data=df)
st.pyplot(fig)

# ----------- Feature Engineering -----------

df["high_rating"] = df["rate"].apply(lambda x: 1 if x >= 4 else 0)

# ----------- Prediction Model -----------

st.header("Restaurant Success Prediction")

# Define success
df["is_successful"] = df.apply(lambda x: 1 if (x["rate"] >= 4 and x["votes"] > 200) else 0, axis=1)

# Select features
model_df = df[["approx_cost(for two people)", "listed_in(type)", "is_successful"]].dropna()

# Encode categorical variable
le = LabelEncoder()
model_df["listed_in(type)"] = le.fit_transform(model_df["listed_in(type)"])

X = model_df[["approx_cost(for two people)", "listed_in(type)"]]
y = model_df["is_successful"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

st.subheader("Model Accuracy")
st.write(acc)

# ----------- User Prediction -----------

st.subheader("Predict Restaurant Success")

cost = st.number_input("Approx Cost for Two", 100, 2000, 500)

rest_type = st.selectbox(
    "Restaurant Type",
    df["listed_in(type)"].unique()
)

rest_type_encoded = le.transform([rest_type])[0]

if st.button("Predict"):
    result = model.predict([[cost, rest_type_encoded]])

    if result[0] == 1:
        st.success("This restaurant is likely to be successful")
    else:
        st.error("This restaurant may not be successful")