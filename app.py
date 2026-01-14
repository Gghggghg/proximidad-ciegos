import streamlit as st
import requests
import json

st.title("Ventana online — resultados del detector")

# Ejemplo: leer JSON desde GitHub raw o API local
url = "https://raw.githubusercontent.com/tuusuario/turepo/main/results.json"
resp = requests.get(url)
data = resp.json()

st.json(data)  # muestra el JSON completo

# Visualización rápida
for det in data["detections"]:
    st.write(f"Clase: {det['class']} | Distancia: {det['distance']}m | Zona: {det['zone']}")

if data["alarm"]:
    st.error("🚨 ¡Alarma activa!")



