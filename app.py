import streamlit as st

from utils import classify_image, pipe


# Interfaz de usuario
st.title("Generador y Clasificador de Imágenes")

with st.container():
    col1, col2 = st.columns(2)

    with col1:

        prompt = st.text_input("Descripción de la imagen:")
        if st.button("Generar"):
         
            image = pipe(prompt, guidance_scale=7.5).images[0]
            st.image(image)
            st.session_state.image = image

    with col2:
        if "image" in st.session_state:
            st.image(st.session_state.image)
            prediction = classify_image(st.session_state.image)
            st.write("Predicción:", prediction)

