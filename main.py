import streamlit as st
from PIL import Image
from predict import predict


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("VisualFeast Simple Image Classification App")
st.write("")
st.write("")

file_up = st.file_uploader("Upload an image")

if file_up is None:
    image = 'image/dog.jpg'
    img = Image.open(image)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")

    labels, ret = predict(image)
    st.success('successful prediction')
    st.write("Prediction (index, name)", labels, ",   Score: ", ret)
    st.write("")

else:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels,ret = predict(file_up)

    st.success('successful prediction')
    st.write("Prediction (index, name)", labels, ",   Score: ", ret)
    st.write("")



