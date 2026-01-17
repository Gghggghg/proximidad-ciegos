

# Tutorial de Instalación en Windows  
**Proyecto:** *AnomalyDetector.py*

---

## 1. Requisitos Previos
- **Sistema operativo:** Windows 10 o superior.  
- **Python:** versión 3.9 o superior (recomendado 3.10).  
- **Cámara web:** integrada o externa.  
- **Conexión a internet:** solo para instalar librerías.  

---

## 2. Instalación de Python
1. Descarga Python desde la página oficial: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/).  
2. Durante la instalación, marca la opción **“Add Python to PATH”**.  
3. Verifica la instalación abriendo **CMD** y escribiendo:  
   ```
   python --version
   ```

---

## 3. Crear Carpeta y Entorno Virtual
1. Abre **CMD** o **PowerShell**.  
2. Crea una carpeta para el proyecto:  
   ```
   mkdir AnomalyDetector
   cd AnomalyDetector
   ```
3. Crea un entorno virtual:  
   ```
   python -m venv venv
   ```
4. Activa el entorno:  
   ```
   venv\Scripts\activate
   ```

---

## 4. Instalar Dependencias
Con el entorno activado, instala las librerías necesarias:

```
pip install opencv-python
pip install numpy
pip install streamlit
pip install streamlit-webrtc
pip install ultralytics
pip install pygame
```

---

## 5. Archivos del Proyecto
En la carpeta del proyecto deben estar:  
- **`AnomalyDetector.py`** → archivo principal con el código.  
- **`yolov8s.pt`** → modelo YOLO (ya incluido en las carpetas).  
- **Audios WAV**:  
  - `beep.wav` (pitido inicial).  
  - `0.wav, 1.wav, 2.wav, 3.wav, 4.wav, 5.wav` (conteo de personas).  



---

## 6. Ejecutar el Proyecto
1. Abre la terminal en la carpeta del proyecto.  
2. Ejecuta:  
   ```
   streamlit run AnomalyDetector.py
   ```
3. Se abrirá automáticamente en tu navegador en:  
   ```
   http://localhost:8501
   ```

---

## 7. Crear un .exe en Windows
Para que solo haga doble clic y abra el programa:

### 1. Instalar PyInstaller
Con tu entorno virtual activado:
```
pip install pyinstaller
```

### 2. Generar el ejecutable
En la carpeta del proyecto, ejecuta:
```
pyinstaller --onefile --noconsole AnomalyDetector.py
```

- **`--onefile`**: empaqueta todo en un único `.exe`.  
- **`--noconsole`**: evita que se abra la ventana negra de consola.  
- El resultado estará en la carpeta `dist/` como:
  ```
  AnomalyDetector.exe
  ```

### 3. Incluir archivos necesarios
Copia junto al `.exe`:
- `yolov8s.pt`  
- `beep.wav`  
- `0.wav, 1.wav, 2.wav, 3.wav, 4.wav, 5.wav`  

 El `.exe` debe estar en la misma carpeta que estos archivos para que se encuentren correctamente.

### 4. Ejecutar
- Haz doble clic en `AnomalyDetector.exe`.  
- Se abrirá automáticamente el navegador en:
  ```
  http://localhost:8501
  ```

---

## 8. Validación
Al ejecutar, el sistema debe:  
- Detectar personas y vehículos en tiempo real.  
- Estimar distancias usando el modelo pinhole.  
- Emitir pitidos progresivos según proximidad.  
- Anunciar el número de personas detectadas (0–5).  
- Mostrar video anotado con colores según distancia.  

