import streamlit as st
from image_classification import teachable_machine_classification
st.title("Diseases Classification on Potato's Leaf")
st.header("Brain Tumor MRI Classification Example")
st.text("Upload a Potato leaf Image for Diseases classification as Late_Blight or Early_Blight  or Healthy")


uploaded_file = st.file_uploader("Choose a Potato's Leaf Image ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded MRI.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        label = teachable_machine_classification(image, 'potatoes.h5')
        if label == 0:
            st.write("The Potatoe's leaf is Early_Blight")
        elif label==1:
            st.write("The Potatoe's leaf is Late_Blight")
        else:
            st.write("The Potato leaf  is healthy")
