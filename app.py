
# ============================
# INICIO: Librerías
# - Orden de carga intencional:
#   1) Computación y tiempo (numpy, time)
#   2) Visión (cv2)
#   3) Web (streamlit)
#   4) Modelo IA (ultralytics.YOLO)
#   5) Audio (pygame), con fallback seguro
#   Motivo: mantener inicialización predecible y mensajes claros en caso de fallos.
# ============================
import cv2
import time
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ============================
# INPUT: Variables y parámetros iniciales
# Propósito pedagógico:
# - Declarar explícitamente cada recurso (audio, IA, cámara, focal)
# - Documentar supuestos y límites (clases, alturas, umbrales, NA)
# - Asegurar trazabilidad para no videntes mediante resumen textual consistente.
# ============================

# --- Audio ---
# - AUDIO_ENABLED controla si se intentará reproducir pitidos (accesibilidad auditiva).
# - BEEP_FILE es el recurso de sonido (mono, corto, no intrusivo).
# - La inicialización de pygame usa parámetros conservadores para baja latencia.
# - Si falla cualquier paso (archivo ausente, mixer no disponible), se desactiva audio
#   sin romper el flujo principal (detección y visualización continúan).
AUDIO_ENABLED = True
BEEP_FILE = "beep.wav"
try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    beep_sound = pygame.mixer.Sound(BEEP_FILE)
except Exception:
    AUDIO_ENABLED = False
    beep_sound = None

# --- Modelo YOLO ---
# - yolov8s.pt: modelo liviano para tiempo real en CPU; ajusta a "n" o "m" según hardware.
# - verbose se controla más adelante para no llenar la interfaz.
model = YOLO("yolov8s.pt")

# --- Clases relevantes con alturas arbitrarias de ejemplo ---
# - size_m: dimensión real aproximada para inferir distancia (pinhole).
# - axis: eje de medición preferente (height por estabilidad en pedestres).
# - Nota: Son valores pedagógicos; en producción calibrar por clase y cámara.
CLASS_REAL_SIZE = {
    "person": {"size_m": 1.65, "axis": "height"},
    "car": {"size_m": 1.80, "axis": "height"},
    "bus": {"size_m": 2.50, "axis": "height"},
    "truck": {"size_m": 2.50, "axis": "height"},
    "motorcycle": {"size_m": 1.60, "axis": "height"},
    "bicycle": {"size_m": 1.60, "axis": "height"},
}
TARGET_CLASSES = set(CLASS_REAL_SIZE.keys())

# --- Ajuste focal ---
# - STATE["focal_px"] proviene de una calibración previa (pinhole).
# - Mantener en estado global para trazabilidad y actualización futura.
STATE = {"focal_px": 327.62}

# --- Control de audio ---
# - last_beep_time evita saturación auditiva. Intervalos dependen de distancia y tipo.
last_beep_time = 0.0

# ============================
# DEFINIR OPERACIONES: Funciones auxiliares
# Diseño:
# - Cada función tiene responsabilidad única y comentarios de entrada/salida.
# - No lanzan excepciones (devuelven None o silencian), para continuidad pedagógica.
# ============================

def pick_axis_size(box, axis="height"):
    """
    Devuelve el tamaño del eje solicitado a partir de una caja (x1, y1, x2, y2).

    Parámetros:
    - box: tupla (x1, y1, x2, y2) en píxeles
    - axis: "height" o "width"

    Retorna:
    - int tamaño en px, mínimo 1 para evitar división por cero.

    Justificación:
    - height es más estable para personas. width puede variar por pose/inclinación.
    """
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w if axis == "width" else h

def estimate_distance(size_px, H_real_m, focal_px):
    """
    Calcula distancia aproximada con cámara pinhole:
    D ≈ (H_real_m * focal_px) / size_px

    Parámetros:
    - size_px: tamaño en píxeles del eje elegido
    - H_real_m: altura (o ancho) real mapeada a la clase
    - focal_px: distancia focal efectiva en píxeles (calibrada)

    Retorna:
    - float con metros, o None si no es posible calcular (evita fallos).
    """
    if focal_px <= 0 or size_px <= 0:
        return None
    return (H_real_m * focal_px) / size_px

def beep_progresivo(D, tipo):
    """
    Genera pitido progresivo según distancia y tipo de objeto.
    Políticas:
    - Personas: umbrales más sensibles (proximidad crítica a 2 m).
    - Vehículos: umbrales más amplios (proximidad crítica a 3 m).
    - Se limita la frecuencia de pitidos con last_beep_time.

    Seguridad:
    - Si AUDIO_ENABLED es False o D es None, no hace nada.
    - Cualquier excepción en el mixer se silencia.

    Accesibilidad:
    - Volumen y ritmo aumentan cuando el objeto se acerca, para clara percepción auditiva.
    """
    global last_beep_time
    if not AUDIO_ENABLED or D is None:
        return
    now = time.time()
    if tipo == "person":
        if D > 10: return
        if D > 5: interval, vol = 1.0, 0.2
        elif D > 2: interval, vol = 0.6, 0.5
        else: interval, vol = 0.3, 0.9
    else:
        if D > 10: return
        if D > 6: interval, vol = 1.0, 0.3
        elif D > 3: interval, vol = 0.6, 0.6
        else: interval, vol = 0.3, 0.9
    if now - last_beep_time < interval:
        return
    last_beep_time = now
    try:
        if beep_sound is not None:
            beep_sound.set_volume(vol)
            beep_sound.play()
    except Exception:
        pass

def auto_camera_index(max_test=3):
    """
    Detecta automáticamente la primera cámara disponible.
    - Prueba índices [0..max_test-1].
    - Retorna el índice o None si no hay cámara.

    Motivo:
    - Facilitar despliegue en distintos equipos sin configuración manual.
    """
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

# ============================
# PROCESO DE LAS OPERACIONES: Detección y cálculo
# - Enfocado en mantener tiempo real y trazabilidad textual.
# - Políticas de descarte:
#   * conf < 0.70: no se contabiliza ni se muestra (pedagógicamente coherente).
#   * cls fuera de TARGET_CLASSES: se ignora (evita ruido).
# - Reporte:
#   * Siempre mostrar distancias aceptadas; si se descarta no se reporta (consistencia).
# - Alarmas:
#   * Persona: D < 2 m, o caja dominante (ratio_w > 0.45 o ratio_h > 0.60).
#   * Vehículo: D < 3 m, o ratio_h > 0.60 (ocupa gran parte de pantalla).
# ============================

def procesar(frame, frame_count):
    """
    Ejecuta inferencia, calcula distancias, decide color y arma resumen textual.

    Retorna:
    - annotated: imagen con cajas y etiquetas coloreadas
    - resumen_text: texto multi-línea con conteos, distancias y estado de alarma
    """
    results = model(frame, imgsz=320, verbose=False)[0]
    annotated = frame.copy()
    resumen = []
    fpx = STATE["focal_px"]
    h_img, w_img = frame.shape[:2]
    alarma = False
    count_personas, count_vehiculos = 0, 0
    distancias_personas, distancias_vehiculos = [], []

    # ============================
    # PROCESO DE DECISIÓN: Condiciones y bucles
    # ============================
    for box in results.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)

        # 1) Filtro por confianza y clase de interés
        if conf < 0.70 or cls not in TARGET_CLASSES:
            # No se reporta para evitar contradicciones en salida
            continue

        # 2) Calcular distancia con pinhole, usando eje recomendado por clase
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        axis = CLASS_REAL_SIZE[cls]["axis"]
        H_real = CLASS_REAL_SIZE[cls]["size_m"]
        size_px = pick_axis_size((x1, y1, x2, y2), axis)
        D = estimate_distance(size_px, H_real, fpx)

        # 3) Calcular zona horizontal (izquierda/centro/derecha) para orientación auditiva/textual
        cx = (x1 + x2) // 2
        if cx < w_img / 3:
            zona = "izquierda"
        elif cx < 2 * w_img / 3:
            zona = "centro"
        else:
            zona = "derecha"

        # 4) Ratios relativos (tamaño de caja respecto a la imagen)
        #    Ayudan a detectar ocupación dominante aunque D sea None (por geometría o oclusión).
        box_width = x2 - x1
        ratio_w = box_width / w_img
        box_height = y2 - y1
        ratio_h = box_height / h_img

        # 5) Decisiones por tipo
        if cls == "person":
            count_personas += 1
            if D is not None:
                distancias_personas.append(D)
                beep_progresivo(D, "person")
            # Persona: alarma si D < 2 m o si ocupa área significativa
            if (D is not None and D < 2.0) or ratio_w > 0.45 or ratio_h > 0.60:
                alarma = True
        else:
            count_vehiculos += 1
            if D is not None:
                distancias_vehiculos.append(D)
                # Nota: usamos "vehiculo" para la lógica interna de audio (no afecta visual)
                beep_progresivo(D, "vehiculo")
            # Vehículo: alarma si D < 3 m o si ocupa vertical dominante
            if (D is not None and D < 3.0) or ratio_h > 0.60:
                alarma = True

        # 6) Color según proximidad (verde/amarillo/naranja/rojo)
        #    - Rojo: crítico, umbral por tipo
        #    - Naranja: media proximidad
        #    - Amarillo: lejana pero relevante
        #    - Verde: seguro
        color = (0, 255, 0)
        if D is not None:
            if (cls == "person" and D < 2) or (cls != "person" and D < 3):
                color = (0, 0, 255)          # rojo
            elif (cls == "person" and D < 5) or (cls != "person" and D < 6):
                color = (0, 165, 255)        # naranja
            elif D < 10:
                color = (0, 255, 255)        # amarillo

        # ============================
        # RESULTADOS parciales (visual + texto)
        # ============================
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f} | {D if D is not None else 'NA'}m | {zona}"
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        resumen.append(label)

    # ============================
    # DECISIÓN: RESULTADOS finales (texto accesible)
    # - Siempre incluir conteos por tipo.
    # - Listar distancias aceptadas (no incluir descartadas).
    # - Mensaje de alarma dinámico (solo si alguna condición se activó).
    # ============================
    if alarma:
        resumen.append("🚨 ALARMA: objeto o señal cerca")
    resumen.append(f"👥 Personas: {count_personas}")
    if distancias_personas:
        resumen.append("   Distancias: " + ", ".join(f"{d:.2f}m" for d in distancias_personas))
    resumen.append(f"🚗 Vehículos: {count_vehiculos}")
    if distancias_vehiculos:
        resumen.append("   Distancias: " + ", ".join(f"{d:.2f}m" for d in distancias_vehiculos))
    if not resumen:
        resumen = ["❓ Nada detectado"]

    return annotated, "\n".join(resumen)


# ============================
# OUTPUT: Visualización y control en web (STREAMLIT)
# Estructura:
# - header: contexto de cámara del servidor
# - placeholders: imagen y texto con actualización en bucle
# - control: checkbox para iniciar/detener
# Robustez:
# - Mensajes de error/advertencia claros (no bloquean app completa)
# - Release de recursos al finalizar
# ============================

st.header("Cámara del servidor (compartida)")
frame_placeholder = st.empty()
text_placeholder = st.empty()

# Detección automática de cámara (entorno local/servidor)
cam_index = auto_camera_index()
if cam_index is None:
    st.error("⚠️ No se encontró ninguna cámara disponible")
else:
    cap = cv2.VideoCapture(cam_index)
    run = st.checkbox("Iniciar cámara")

    frame_count = 0
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ No se pudo acceder a la cámara")
            break

        # OUTPUT EN LA WEB
        annotated, resumen = procesar(frame, frame_count)
        # Convertimos BGR->RGB para visualización correcta en web
        frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        # Texto accesible con decisiones y distancias
        text_placeholder.text(resumen)

        frame_count += 1

    # Liberación de recursos (cámara y ventanas)
    cap.release()
    cv2.destroyAllWindows()

