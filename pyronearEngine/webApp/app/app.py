import streamlit as st
from PIL import Image
import numpy as np
from PyronearEngine.inference.pyronearPredict import PyronearPredictor

pyronearPredictor = PyronearPredictor("model/pyronearModel.pth")

st.image(Image.open('logo//pyronear-logo-dark.png'), use_column_width=True)
st.title("PyroNear: wildfire early detection")
st.header('The Pyronear detector has been trained to detect a fire outbreak in a forest, \
           please use it only in this context')

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Analyzed image', use_column_width=True)

    res = pyronearPredictor.predict(image)

    if res[0] > 0.5:
        st.write('No fire detected :evergreen_tree:')

    else:
        st.write('Fire detected :fire:')
