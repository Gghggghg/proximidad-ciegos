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
BEEP_FILE = "beep.wav"         # Archivo de pitido básico para alertas
AUDIO_ENABLED = True           # Bandera para habilitar/deshabilitar audio
try:
    import pygame              # Librería para reproducir sonidos
    # Inicialización conservadora del mixer para baja latencia y estabilidad
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    beep_sound = pygame.mixer.Sound(BEEP_FILE)  # Carga del pitido
except Exception:
    # Fallback seguro: si falla el audio, no se rompe el flujo principal
    AUDIO_ENABLED = False
    beep_sound = None

# ============================
# Audios numéricos (0.wav ... 5.wav)
# ============================
NUM_SOUNDS = {}
for i in range(0, 6):  # Bucle for: carga audios de números del 0 al 5 (incluye 0)
    try:
        NUM_SOUNDS[i] = pygame.mixer.Sound(f"{i}.wav")
    except Exception:
        NUM_SOUNDS[i] = None   # Si no existe el archivo, se asigna None para evitar excepciones

# Variables de estado para controlar repetición de audios
last_count = 0           # Último número de personas detectadas (evita repetir audio cada frame)
last_audio_time = 0.0    # Último tiempo en que se reprodujo audio (cooldown)

# ============================
# Modelo YOLO y parámetros de distancia
# ============================
model = YOLO("yolov8n.pt")  # Carga del modelo liviano YOLOv8 (adecuado para CPU)

# Tamaños reales aproximados de objetos para cálculo de distancia con pinhole
# - size_m: altura real aproximada en metros
# - axis: eje preferente para medir en píxeles (height más estable en personas)
CLASS_REAL_SIZE = {
    "person": {"size_m": 1.65, "axis": "height"},
    "car": {"size_m": 1.80, "axis": "height"},
    "bus": {"size_m": 2.50, "axis": "height"},
    "truck": {"size_m": 2.50, "axis": "height"},
    "motorcycle": {"size_m": 1.60, "axis": "height"},
    "bicycle": {"size_m": 1.60, "axis": "height"},
}
TARGET_CLASSES = set(CLASS_REAL_SIZE.keys())  # Clases relevantes para el sistema
STATE = {"focal_px": 333.33}                  # Focal calibrada en píxeles (pinhole)
last_beep_time = 0.0                          # Control de pitidos progresivos (cooldown)
first_frame = True                            # Bandera para pitido inicial al recibir primer frame

# Umbral configurable para ancho de persona (proporción del ancho de la imagen)
UMBRAL_ANCHO_PERSONA = 0.45

# ============================
# Funciones auxiliares
# ============================
def pick_axis_size(box, axis="height"):
    """
    Devuelve el tamaño en píxeles de la caja detectada según el eje elegido.
    - box: coordenadas (x1, y1, x2, y2)
    - axis: 'height' o 'width'
    Uso:
    - height es más estable para personas; width puede variar por pose/inclinación.
    """
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)  # max(1, ...) evita división por cero
    h = max(1, y2 - y1)
    return w if axis == "width" else h

def estimate_distance(size_px, H_real_m, focal_px):
    """
    Calcula distancia aproximada usando el modelo pinhole:
    D ≈ (H_real_m * focal_px) / size_px
    - size_px: tamaño en píxeles del eje elegido (alto o ancho)
    - H_real_m: altura real aproximada del objeto (metros)
    - focal_px: distancia focal efectiva en píxeles (calibrada)
    Retorna:
    - distancia en metros (float) o None si no es posible calcular
    """
    if focal_px <= 0 or size_px <= 0:
        return None
    return (H_real_m * focal_px) / size_px

def beep_progresivo(D, tipo):
    """
    Genera pitidos progresivos según distancia y tipo de objeto.
    Estructuras de control:
    - if: define intervalos de cooldown por tipo y distancia
    - control de saturación con last_beep_time
    Seguridad:
    - Si AUDIO_ENABLED es False o D es None, no hace nada.
    """
    global last_beep_time
    if not AUDIO_ENABLED or D is None:
        return False
    now = time.time()
    # Umbrales diferenciados por tipo (person vs vehículo)
    if tipo == "person":
        if D > 10: return False
        if D > 5: interval = 1.0
        elif D > 2: interval = 0.6
        else: interval = 0.3
    else:
        if D > 10: return False
        if D > 6: interval = 1.0
        elif D > 3: interval = 0.6
        else: interval = 0.3
    # Cooldown: evita reproducir pitidos demasiado seguidos
    if now - last_beep_time < interval:
        return False
    last_beep_time = now
    try:
        if beep_sound is not None:
            beep_sound.play()
    except Exception:
        # Silencia cualquier error del mixer para no interrumpir el flujo
        pass
    return True

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
    global last_count, last_audio_time
    results = model(frame, imgsz=480, verbose=False)[0]  # Inferencia YOLO
    annotated = frame.copy()
    fpx = STATE["focal_px"]
    h_img, w_img = frame.shape[:2]
    count_personas = 0  # Contador de personas detectadas en el frame

    # Bucle for: recorre todas las detecciones del modelo
    for box in results.boxes:
        cls = model.names[int(box.cls)]   # Nombre de clase detectada
        conf = float(box.conf)            # Confianza de la detección
        # Filtro por confianza y clase relevante (estructura de control if)
        if conf < 0.70 or cls not in TARGET_CLASSES:
            continue  # Descarta detecciones poco confiables o irrelevantes

        # Coordenadas de la caja y cálculo de distancia con pinhole
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        axis = CLASS_REAL_SIZE[cls]["axis"]
        H_real = CLASS_REAL_SIZE[cls]["size_m"]
        size_px = pick_axis_size((x1, y1, x2, y2), axis)
        D = estimate_distance(size_px, H_real, fpx)

        # Zona horizontal (izquierda, centro, derecha) para orientación textual/auditiva
        cx = (x1 + x2) // 2
        if cx < w_img / 3:
            zona = "izquierda"
        elif cx < 2 * w_img / 3:
            zona = "centro"
        else:
            zona = "derecha"

        # Proporción de ancho de la caja respecto al ancho de la imagen
        box_width = x2 - x1
        ratio_w = box_width / w_img

        # Contador de personas (estructura de control if)
        if cls == "person":
            count_personas += 1

        # Umbral de ancho para personas: si ocupa gran parte del ancho, marcar crítico
        if cls == "person" and ratio_w > UMBRAL_ANCHO_PERSONA:
            color = (0, 0, 255)  # rojo crítico
            label = f"{cls} {conf:.2f} | {D:.2f}m | {zona} | ancho!"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Pitido progresivo para persona en umbral de ancho
            beep_progresivo(D, "person")
            continue  # Salta al siguiente objeto (estructura de control)

        # Colores según distancia (estructura de control if con umbrales)
        color = (0, 255, 0)  # verde seguro por defecto
        if D is not None:
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

        # Pitido progresivo según tipo (persona vs vehículo)
        beep_progresivo(D, cls if cls == "person" else "vehiculo")

    # Reproducir audio según número de personas con cooldown (estructura de control if)
    now = time.time()
    if count_personas != last_count and 1 <= count_personas <= 5:
        if now - last_audio_time >= 1.0:  # cooldown de 1 segundo
            snd = NUM_SOUNDS.get(count_personas)
            if snd is not None:
                snd.play()
            last_audio_time = now
        last_count = count_personas

    # Reproducir cuando se pierde la detección (cero) con cooldown
    if last_count > 0 and count_personas == 0:
        if now - last_audio_time >= 1.0:  # cooldown de 1 segundo
            snd = NUM_SOUNDS.get(0)
            if snd is not None:
                snd.play()
            last_audio_time = now
        last_count = 0

    return annotated  # Imagen anotada lista para mostrar en la interfaz

# ============================
# VideoTransformer (WebRTC)
# ============================
class VideoProcessor(VideoTransformerBase):
    """
    Clase que transforma cada frame recibido por WebRTC:
    - Convierte el frame a ndarray (BGR).
    - Llama a 'procesar' para anotar y generar audio.
    - Emite un pitido inicial de 2 segundos al recibir el primer frame.
    """
    def transform(self, frame):
        global first_frame
        img = frame.to_ndarray(format="bgr24")
        annotated = procesar(img)

        # Pitido inicial de 2 segundos al recibir la primera imagen (estructura de control if)
        if first_frame:
            first_frame = False
            try:
                if beep_sound is not None:
                    beep_sound.play()
                    time.sleep(2)
                    beep_sound.stop()
            except Exception:
                # Silencia cualquier error del mixer
                pass

        return annotated

# ============================
# Interfaz Streamlit (WebRTC)
# ============================
st.title("Servidor Detector con YOLO")

# Componente WebRTC:
# - Captura video del dispositivo (cámara trasera si está disponible).
# - Aplica el VideoProcessor para anotar y gestionar audio.
# - async_transform=True permite procesamiento asíncrono para mantener fluidez.
webrtc_streamer(
    key="detector_video",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {"facingMode": "environment"},  # cámara trasera
        "audio": False                           # audio deshabilitado en el stream (audio local via pygame)
    },
    async_transform=True,
)

