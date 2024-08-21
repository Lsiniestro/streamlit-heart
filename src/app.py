from pickle import load

import streamlit as st


model = load(open("../models/model_xgb.pkl", "rb"))

class_dict = {"0":'No Heart Disease', "1":'Heart Disease'}


st.title("Heart Disease Prediction")


val1 = st.slider("Age", min_value = 10, max_value = 100, step = 1)

val2 = st.slider("Ejection fraction", min_value = 10, max_value = 80, step = 1)

val3 = st.slider("Serum creatinine", min_value = 0.5, max_value = 10.0, step = 0.01)

val4 = st.slider("Serum Sodium", min_value = 100, max_value = 150, step = 1)


if st.button("Predict"):

    prediction = str(model.predict([[val1, val2, val3, val4]])[0])

    pred_class = class_dict[prediction]

    st.write("Prediction:", pred_class)