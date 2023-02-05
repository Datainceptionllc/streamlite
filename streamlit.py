import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Load the data
@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train the ML model
@st.cache
def train_model(df):
    X = df.drop("class", axis=1)
    y = df["class"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Make predictions
def predict(model, sequence):
    prediction = model.predict([sequence])[0]
    return prediction

# Create the streamlit app
def main():
    st.title("Sequence Prediction App")
    file_path = st.file_uploader("Upload the data file", type=["csv"])
    if file_path is not None:
        df = load_data(file_path)
        model = train_model(df)
        sequence = st.text_input("Enter the sequence")
        if st.button("Input"):
            st.write("You entered:", sequence)
        if st.button("Predict"):
            prediction = predict(model, sequence)
            st.write("The predicted class is:", prediction)

if __name__ == "__main__":
    main()
