import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import plotly.express as px

# Configuracion de pagina

st.set_page_config(
    page_title="Iris Species Classification",
    layout="wide"
)

st.title("Iris Species Classification")
st.write("Machine Learning Project - Data Mining")

# Cargamos los datos
df = pd.read_csv("Iris.csv")

# Removemos la columna Id existente
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

X = df.drop("Species", axis=1)
y = df["Species"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Modelo
model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

st.subheader("Model Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# User input
st.subheader("Predict Iris Species")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width", 2.0, 5.0, 3.0)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Boton de prediccion
if st.button("Predict Species"):

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(input_data)

    st.success(f"Predicted Species: {prediction[0]}")

# Diagrama
fig = px.scatter_3d(
    df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    z="PetalLengthCm",
    color="Species",
    size="PetalWidthCm"
)

st.plotly_chart(fig, use_container_width=True)

# Histograma
st.subheader("Feature Distribution")

feature = st.selectbox(
    "Select Feature",
    X.columns
)

fig2 = px.histogram(
    df,
    x=feature,
    color="Species",
    barmode="overlay"
)

st.plotly_chart(fig2, use_container_width=True)