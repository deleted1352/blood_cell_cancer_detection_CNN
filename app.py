import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Blood Cancer Detector", page_icon="🩸")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('blood_cancer_model.h5')
    return model

model = load_model()

# labels
CLASS_NAMES = ['Basophil', 'Eosinophil', 'Lymphocytes', 'Monocytes', 'Neutrophil']

# 3. App Header
st.title("Blood Cell Cancer Detection")
st.markdown("""
Upload a microscopic image of blood cells to analyze for potential cancer indicators.
This model uses a **VGG-19** architecture to classify images into four categories.
""")

# upload
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("---")
    st.write("### Analysis Result")
    
    with st.spinner('Analyzing'):
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        
        if img_array.shape[-1] != 3:
            img = img.convert('RGB')
            img_array = np.array(img) / 255.0
            
        img_array = np.expand_dims(img_array, axis=0)

        # prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        result = CLASS_NAMES[np.argmax(score)]
        confidence = 100 * np.max(score)

        # display
        if "Malignant" in result:
            st.error(f"Prediction: **{result}**")
        else:
            st.success(f"Prediction: **{result}**")
            
        st.write(f"**Confidence Level:** {confidence:.2f}%")

# medical ai disclaimer
st.info("**Disclaimer:** This tool is for educational/research purposes only. It is not a substitute for professional medical diagnosis.")