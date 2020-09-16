import streamlit as st 
import torchvision
import torch
from torch import nn
from PIL import Image
from torchvision import transforms
from pyronearPredict import PyronearPredictor

pyronearPredictor = PyronearPredictor()

st.title("PyroNear: wildfire early detection")
st.header('The Pyronear detector has been trained to detect a fire outbreak in a forest, please use it only in this context')

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Analyzed image', use_column_width=True)
    st.write("")
    res = pyronearPredictor.predict(image)
    st.write(res)
