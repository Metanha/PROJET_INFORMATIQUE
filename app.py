import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import os
import gdown

model_path1 = "models/modele_genre.h5"
model_path2 = "models/modele_age.h5"

if not os.path.exists(model_path1):
    # Lien Drive : https://drive.google.com/file/d/1uN7IgmON0azSTR-5SvE4TZ9-lJ7wDZj9/view?usp=sharing
    file_id = "1uN7IgmON0azSTR-5SvE4TZ9-lJ7wDZj9"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

if not os.path.exists(model_path2):
    # Lien Drive : https://drive.google.com/file/d/1XlBFzK0zJceLx8JXZo2bpVvejsdso_pA/view?usp=sharing
    file_id1 = "1XlBFzK0zJceLx8JXZo2bpVvejsdso_pA"
    url1 = f"https://drive.google.com/uc?id={file_id1}"
    gdown.download(url1, model_path1, quiet=False)

from tensorflow.keras.models import load_model
genre_model = load_model(model_path12)
age_model = load_model(model_path)

# Chargement du modèle
dossier="F:/FORMATIONS/MASTER/IA/1ere annee/PROJET INFORMATIQUE/App/"
#@st.cache_resource
#def load_models():
    
#  genre_model = tf.keras.models.load_model(dossier+"model/modele_genre.h5")
#  age_model = tf.keras.models.load_model(dossier+"model/modele_age.h5")
#  return genre_model, age_model

#genre_model, age_model = load_models()

AGE_CLASSES = [f"{i}-{i+4} ans" for i in range(0, 100, 5)] + ["100+ ans"]

# Fonction de prétraitement
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Interface Streamlit
st.title("Détection du Genre et de l'Âge à partir d'une Image")

uploaded_file = st.file_uploader("Choisissez une image de visage 👇", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image chargée", use_column_width=True)

    with st.spinner("Prédiction en cours..."):
        img = preprocess_image(image)
        genre_pred = genre_model.predict(img)[0][0]
        genre = "Homme" if genre_pred >= 0.5 else "Femme"
        #pred = model.predict(img)


        age_logits = age_model.predict(img)[0]
        age_index = np.argmax(age_logits)
        age_tranche = AGE_CLASSES[age_index]


        st.success(f"Genre prédit : **{genre}**")
        st.success(f"Âge estimé : **{age_tranche}**")

