import streamlit as st
import numpy as np
from model import person_detection
from model import delivery_company
from tensorflow.keras.preprocessing import image


def predict(img):
    original_image = image.load_img(img,target_size=(224,224))
    person_found_msg = person_detection(original_image)

    if person_found_msg == "Person found":
      st.write('Human Detected')
      predicted_company_name = delivery_company(original_image)
      st.write(predicted_company_name)
    elif person_found_msg == ("person not found"):
      st.write('No Human Detected')

uploaded_file = st.file_uploader("Choose a file", type=['jpg', 'png'])

if uploaded_file is not None:
    predict(uploaded_file)

