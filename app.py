import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Restaurant Analytics Dashboard")

df = pd.read_csv("restaurant_analytics.csv")

st.write("Dataset Preview")
st.write(df.head())

df["rate"] = df["rate"].str.replace("/5","")
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

st.subheader("Restaurant Type Distribution")
fig, ax = plt.subplots()
sns.countplot(x="listed_in(type)", data=df)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
sns.histplot(df["rate"], bins=10)
st.pyplot(fig)