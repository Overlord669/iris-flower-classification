
import streamlit as st
import numpy as np
import joblib
import os
from sklearn.datasets import load_iris

# Load trained model and scaler
with open("iris_knn_model.pkl", "rb") as file:
    model = joblib.load("iris_knn_model.pkl")
with open("iris_scaler.pkl", "rb") as file:
    scaler = joblib.load("iris_scaler.pkl")

iris = load_iris()

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species using our KNN model."
)

st.divider()

# User inputs
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)", min_value=0.1,
        max_value=10.0, value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)", min_value=0.1,
        max_value=10.0, value=3.5
    )

with col2:
    petal_length = st.number_input(
        "Petal Length (cm)", min_value=0.1,
        max_value=10.0, value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)", min_value=0.1,
        max_value=10.0, value=0.2
    )

# Prediction
if st.button("Predict Flower Species", type="primary"):

    features = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    scaled_features = scaler.transform(features)

    prediction = model.predict(scaled_features)[0]

    if isinstance(prediction, (int, np.integer)):
        species = iris.target_names[int(prediction)]
    else:
        species = str(prediction).lower()
        species = species.replace("iris-", "").replace("iris_", "")

    st.success(f"Predicted Flower Species: {species.title()}")

    # Display flower image
    image_files = {
    "setosa": "iris_setosa.jpg",
    "versicolor": "iris_versicolor.jpg",
    "virginica": "iris_virginica.jpg"
}

    if species in image_files:
        image_path = os.path.join(
            "images", image_files[species]
        )

        if os.path.exists(image_path):
            st.image(image_path, caption=species.title())

    # Display prediction probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(scaled_features)[0]
        st.subheader("Prediction Probabilities")

        for name, probability in zip(
            model.classes_, probabilities
        ):
            if isinstance(name, (int, np.integer)):
                label = iris.target_names[int(name)]
            else:
                label = str(name)

            st.write(f"{label.title()}: {probability:.2%}")
            st.progress(float(probability))

st.divider()

st.caption("Developed using Python, Scikit-learn, KNN and Streamlit.")
