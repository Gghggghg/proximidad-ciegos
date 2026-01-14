import streamlit as st
import requests

st.title("Ventana online — resultados del detector")

url = "https://raw.githubusercontent.com/usuario/repositorio/main/results.json"
resp = requests.get(url)

try:
    data = resp.json()
    st.json(data)  # muestra el JSON completo
    for det in data.get("detections", []):
        st.write(f"Clase: {det['class']} | Distancia: {det['distance']}m | Zona: {det['zone']}")
    if data.get("alarm"):
        st.error("🚨 ¡Alarma activa!")
except Exception:
    st.error("⚠️ El archivo no contiene JSON válido")
    st.text(resp.text)  # muestra el contenido crudo para depuración




