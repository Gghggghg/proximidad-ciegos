# ============================
# Librerías utilizadas
# ============================
import cv2                     # OpenCV: captura y manipulación de imágenes y video
import time                    # Manejo de tiempos y cooldowns para audio
import numpy as np             # Operaciones numéricas y matrices
import streamlit as st         # Interfaz web para mostrar resultados
from ultralytics import YOLO   # Modelo de IA YOLO para detección de objetos
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase  # Transmisión WebRTC en navegador

# ============================
# Configuración de audio local
# ============================
BEEP_FILE = "beep.wav"         # Variable que guarda el nombre del archivo de pitido básico
AUDIO_ENABLED = True           # Bandera (True/False) que indica si el audio está habilitado
try:                           # TRY: intenta ejecutar el bloque de código siguiente
    import pygame              # Importa librería para reproducir sonidos
    # Inicialización conservadora del mixer para baja latencia y estabilidad
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    beep_sound = pygame.mixer.Sound(BEEP_FILE)  # Variable que guarda el sonido cargado en memoria
except Exception:              # EXCEPT: si ocurre un error en el bloque TRY, se ejecuta este bloque
    # Fallback seguro: si falla el audio, no se rompe el flujo principal
    AUDIO_ENABLED = False      # Variable se pone en False → desactiva audio
    beep_sound = None          # Variable se pone en None → no hay sonido cargado

# ============================
# Audios numéricos (0.wav ... 5.wav)
# ============================
NUM_SOUNDS = {}                # Diccionario vacío para guardar sonidos numéricos
for i in range(0, 6):          # FOR: bucle que repite desde i=0 hasta i=5
    try:                       # TRY: intenta cargar cada archivo de sonido
        NUM_SOUNDS[i] = pygame.mixer.Sound(f"{i}.wav")  # Cada clave i guarda un sonido (ej: 0.wav, 1.wav...)
    except Exception:          # EXCEPT: si el archivo no existe o hay error
        NUM_SOUNDS[i] = None   # Guarda None para evitar que el programa se rompa

# Variables de estado para controlar repetición de audios
last_count = 0           # Variable que guarda el último número de personas detectadas
last_audio_time = 0.0    # Variable que guarda el último tiempo en que se reprodujo audio (cooldown)

# ============================
# Modelo YOLO y parámetros de distancia
# ============================
model = YOLO("yolov8n.pt")  # Variable que guarda el modelo liviano YOLOv8 cargado

# Diccionario con tamaños reales aproximados de objetos para cálculo de distancia con pinhole
CLASS_REAL_SIZE = {
    "person": {"size_m": 1.65, "axis": "height"},   # persona: altura real 1.65m, se mide por altura
    "car": {"size_m": 1.80, "axis": "height"},      # auto: altura real 1.80m
    "bus": {"size_m": 2.50, "axis": "height"},      # bus: altura real 2.50m
    "truck": {"size_m": 2.50, "axis": "height"},    # camión: altura real 2.50m
    "motorcycle": {"size_m": 1.60, "axis": "height"}, # moto: altura real 1.60m
    "bicycle": {"size_m": 1.60, "axis": "height"},    # bicicleta: altura real 1.60m
}

TARGET_CLASSES = set(CLASS_REAL_SIZE.keys())  # Variable que guarda las clases relevantes (person, car, bus, etc.)
STATE = {"focal_px": 333.33}                  # Diccionario con la focal calibrada en píxeles
last_beep_time = 0.0                          # Variable que guarda el último tiempo en que sonó un pitido
first_frame = True                            # Bandera que indica si es el primer frame recibido

UMBRAL_ANCHO_PERSONA = 0.45                   # Variable que define el umbral de ancho crítico para personas

# ============================
# Funciones auxiliares
# ============================
def pick_axis_size(box, axis="height"):
    """
    Devuelve el tamaño en píxeles de la caja detectada según el eje elegido.
    """
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)  # Variable w: ancho de la caja en píxeles (mínimo 1 para evitar división por cero)
    h = max(1, y2 - y1)  # Variable h: alto de la caja en píxeles (mínimo 1)
    return w if axis == "width" else h  # IF condicional: si axis es "width" devuelve ancho, si no devuelve alto

def estimate_distance(size_px, H_real_m, focal_px):
    """
    Calcula distancia aproximada usando el modelo pinhole.

    Variables:
    - size_px: tamaño en píxeles del objeto detectado (ancho o alto de la caja delimitadora).
    - H_real_m: tamaño real del objeto en metros (por ejemplo, altura de una persona).
    - focal_px: distancia focal de la cámara en píxeles (calibrada previamente).

    Lógica:
    Usando la fórmula del modelo pinhole, la distancia se estima como:
    distancia = (tamaño real * distancia focal) / tamaño en píxeles

    Esto significa que si el objeto parece más pequeño en píxeles, está más lejos, y si es más grande, está más cerca.

    Retorna:
    - La distancia estimada en metros (float) o None si los valores no son válidos.
    """
    if focal_px <= 0 or size_px <= 0:  # IF: si focal o tamaño son inválidos
        return None                    # devuelve None (no se puede calcular)
    return (H_real_m * focal_px) / size_px  # fórmula de distancia → devuelve variable con metros estimados

def beep_progresivo(D, tipo):
    """
    Genera pitidos progresivos según distancia y tipo de objeto.
    """
    global last_beep_time              # Variable global que guarda el último tiempo de pitido
    if not AUDIO_ENABLED or D is None: # IF: si el audio está deshabilitado o no hay distancia calculada
        return False
    now = time.time()                  # Variable now: guarda el tiempo actual
    # IF anidados: definen intervalos según tipo y distancia
    if tipo == "person":
        if D > 10: return False        # demasiado lejos, no suena
        if D > 5: interval = 1.0       # Variable interval: pitido cada 1s si está entre 5 y 10m
        elif D > 2: interval = 0.6     # pitido cada 0.6s si está entre 2 y 5m
        else: interval = 0.3           # pitido cada 0.3s si está a menos de 2m
    else:                              # ELSE: aplica reglas para vehículos
        if D > 10: return False
        if D > 6: interval = 1.0
        elif D > 3: interval = 0.6
        else: interval = 0.3
    if now - last_beep_time < interval:  # IF: controla cooldown, evita pitidos demasiado seguidos
        return False
    last_beep_time = now                 # Actualiza variable con el nuevo tiempo de pitido
    try:                                 # TRY: intenta reproducir el pitido
        if beep_sound is not None:       # IF: solo si hay sonido cargado
            beep_sound.play()
    except Exception:                    # EXCEPT: si falla, ignora error
        pass
    return True                          # Devuelve True si se logró reproducir pitido
# ============================
# Procesamiento de frames
# ============================
def procesar(frame):
    """
    Procesa cada frame:
    - Ejecuta detección con YOLO (IA).
    - Calcula distancias con pinhole.
    - Determina zona horizontal (izquierda/centro/derecha).
    - Aplica colores según proximidad (verde/amarillo/naranja/rojo).
    - Reproduce pitidos progresivos y audios numéricos con cooldown.
    Retorna:
    - annotated: imagen con cajas y etiquetas coloreadas
    """
    global last_count, last_audio_time   # Variables globales: último conteo de personas y último tiempo de audio
    results = model(frame, imgsz=480, verbose=False)[0]  # results: detecciones de YOLO en el frame
    annotated = frame.copy()             # annotated: copia del frame original para dibujar encima
    fpx = STATE["focal_px"]              # fpx: focal calibrada en píxeles (para cálculo de distancia)
    h_img, w_img = frame.shape[:2]       # h_img: alto de la imagen, w_img: ancho de la imagen
    count_personas = 0                   # contador de personas detectadas en este frame

    # FOR: recorre todas las detecciones del modelo YOLO
    for box in results.boxes:
        cls = model.names[int(box.cls)]   # cls: nombre de la clase detectada (ej: "person", "car")
        conf = float(box.conf)            # conf: nivel de confianza de la detección (0 a 1)
        # IF: filtro por confianza y clase relevante
        if conf < 0.70 or cls not in TARGET_CLASSES:
            continue  # Salta detecciones poco confiables o irrelevantes

        # Variables de coordenadas de la caja detectada
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # coordenadas de la caja en píxeles
        axis = CLASS_REAL_SIZE[cls]["axis"]              # axis: eje elegido para medir (height o width)
        H_real = CLASS_REAL_SIZE[cls]["size_m"]          # H_real: altura real aproximada del objeto en metros
        size_px = pick_axis_size((x1, y1, x2, y2), axis) # size_px: tamaño en píxeles de la caja
        D = estimate_distance(size_px, H_real, fpx)      # D: distancia estimada en metros usando pinhole

        # cx: centro horizontal de la caja
        cx = (x1 + x2) // 2
        # IF: determina zona horizontal (izquierda, centro, derecha)
        if cx < w_img / 3:
            zona = "izquierda"
        elif cx < 2 * w_img / 3:
            zona = "centro"
        else:
            zona = "derecha"

        # Variables para proporción de ancho de la caja respecto al ancho de la imagen
        box_width = x2 - x1
        ratio_w = box_width / w_img

        # IF: contador de personas
        if cls == "person":
            count_personas += 1  # suma 1 al contador si la clase es persona

        # IF: umbral de ancho crítico para personas
        if cls == "person" and ratio_w > UMBRAL_ANCHO_PERSONA:
            color = (0, 0, 255)  # rojo crítico
            label = f"{cls} {conf:.2f} | {D:.2f}m | {zona} | ancho!"  # etiqueta con datos
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)    # dibuja rectángulo rojo
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)      # escribe texto encima
            beep_progresivo(D, "person")  # llama función de pitido progresivo
            continue  # Salta al siguiente objeto (no sigue procesando este)

        # IF: colores según distancia
        color = (0, 255, 0)  # verde seguro por defecto
        if D is not None:    # IF: solo si se pudo calcular distancia
            if (cls == "person" and D < 2) or (cls != "person" and D < 3):
                color = (0, 0, 255)      # rojo crítico
            elif (cls == "person" and D < 5) or (cls != "person" and D < 6):
                color = (0, 165, 255)    # naranja medio
            elif D < 10:
                color = (0, 255, 255)    # amarillo lejano

        # Dibujo de caja y etiqueta con distancia y zona
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f} | {D:.2f}m | {zona}" if D else f"{cls} {conf:.2f} | NA | {zona}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Pitido progresivo según tipo (persona o vehículo)
        beep_progresivo(D, cls if cls == "person" else "vehiculo")

    # IF: reproducir audio según número de personas con cooldown
    now = time.time()  # now: tiempo actual en segundos
    if count_personas != last_count and 1 <= count_personas <= 5:
        if now - last_audio_time >= 1.0:  # IF: cooldown de 1 segundo
            snd = NUM_SOUNDS.get(count_personas)  # snd: sonido correspondiente al número de personas
            if snd is not None:           # IF: si existe el sonido cargado
                snd.play()                # reproduce el sonido
            last_audio_time = now         # actualiza tiempo del último audio
        last_count = count_personas       # actualiza último conteo de personas

    # IF: reproducir cuando se pierde la detección (cero personas)
    if last_count > 0 and count_personas == 0:
        if now - last_audio_time >= 1.0:  # IF: espera de 1 segundo
            snd = NUM_SOUNDS.get(0)       # snd: sonido para cero personas
            if snd is not None:           # IF: si existe el sonido
                snd.play()                # reproduce sonido de "0"
            last_audio_time = now         # actualiza tiempo del último audio
        last_count = 0                    # reinicia contador a cero

    return annotated  # Devuelve imagen anotada con cajas y etiquetas


# ============================
# VideoTransformer (WebRTC)
# ============================
class VideoProcessor(VideoTransformerBase):
    """
    Clase que transforma cada frame recibido por WebRTC:
    - Convierte el frame a ndarray (BlueGreenyRed).
    - Llama a 'procesar' para anotar y generar audio.
    - Emite un pitido inicial de 2 segundos al recibir el primer frame.
    """
    def transform(self, frame):
        global first_frame   # Variable global que indica si es el primer frame
        img = frame.to_ndarray(format="bgr24")  # Variable img: convierte frame a matriz de píxeles
        annotated = procesar(img)               # Variable annotated: resultado procesado

        # IF: pitido inicial al recibir la primera imagen
        if first_frame:
            first_frame = False
            try:                                # TRY: intenta reproducir pitido inicial
                if beep_sound is not None:      # IF: solo si hay sonido cargado
                    beep_sound.play()
                    time.sleep(2)               # Espera 2 segundos
                    beep_sound.stop()
            except Exception:                   # EXCEPT: si falla, ignora error
                pass

        return annotated  # Devuelve imagen anotada
# ============================
# Interfaz Streamlit (WebRTC)
# ============================
st.title("Servidor (AnomalyDetector) con YOLO")  # Variable título en la interfaz

# Componente WebRTC:
webrtc_streamer(
    key="detector_video",                       # Variable key: identificador único del stream
    video_processor_factory=VideoProcessor,     # Usa la clase VideoProcessor para procesar video
    media_stream_constraints={
        "video": {"facingMode": "environment"}, # Variable facingMode: cámara trasera
        "audio": False                          # Variable audio: deshabilitado en el stream
    },
    async_transform=True,                       # Variable async_transform: procesamiento asíncrono
)


