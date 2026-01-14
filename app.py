import time
import queue
import numpy as np
import cv2
import av
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

st.set_page_config(page_title="Proximidad accesible", layout="wide")
st.title("Detector de proximidad — cámara del navegador (en vivo)")

# Modelo YOLO (usa yolov8n para menor latencia en Cloud)
model = YOLO("yolov8n.pt")
model.to("cpu")
_ = model.predict(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=320, verbose=False)

CLASS_REAL_SIZE = {
    "person": {"size_m": 1.65, "axis": "height"},
    "car": {"size_m": 1.80, "axis": "height"},
    "bus": {"size_m": 2.50, "axis": "height"},
    "truck": {"size_m": 2.50, "axis": "height"},
    "motorcycle": {"size_m": 1.60, "axis": "height"},
    "bicycle": {"size_m": 1.60, "axis": "height"},
}
TARGET_CLASSES = set(CLASS_REAL_SIZE.keys())
STATE = {"focal_px": 327.62}

def generate_beep(sr=44100, duration=0.12, freq=880):
    t = np.linspace(0, duration, int(sr * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    fade = np.linspace(0, 1, int(sr * 0.02))
    wave[:fade.size] *= fade
    wave[-fade.size:] *= fade[::-1]
    import io, wave as wav
    buf = io.BytesIO()
    with wav.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wave * 32767).astype(np.int16).tobytes())
    return buf.getvalue()

BEEP_WAV = generate_beep()

def pick_axis_size(box, axis="height"):
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    return w if axis == "width" else h

def estimate_distance(size_px, H_real_m, focal_px):
    if focal_px <= 0 or size_px <= 0:
        return None
    return (H_real_m * focal_px) / size_px
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.last_beep_time = 0.0
        self.summary_queue = queue.Queue(maxsize=1)
        self.frame_skip = 1
        self._counter = 0

    def _beep_progresivo(self, D, tipo):
        if D is None:
            return False
        now = time.time()
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
        if now - self.last_beep_time < interval:
            return False
        self.last_beep_time = now
        return True

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        self._counter += 1
        if self._counter % self.frame_skip != 0:
            return frame

        results = model(img, imgsz=320, verbose=False)[0]
        annotated = img.copy()

        resumen = []
        fpx = STATE["focal_px"]
        h_img, w_img = img.shape[:2]
        alarma = False
        count_personas, count_vehiculos = 0, 0
        distancias_personas, distancias_vehiculos = [], []
        trigger_beep = False

        for box in results.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            if conf < 0.70 or cls not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            axis = CLASS_REAL_SIZE[cls]["axis"]
            H_real = CLASS_REAL_SIZE[cls]["size_m"]
            size_px = pick_axis_size((x1, y1, x2, y2), axis)
            D = estimate_distance(size_px, H_real, fpx)

            cx = (x1 + x2) // 2
            if cx < w_img / 3:
                zona = "izquierda"
            elif cx < 2 * w_img / 3:
                zona = "centro"
            else:
                zona = "derecha"

            box_width = x2 - x1
            ratio_w = box_width / w_img
            box_height = y2 - y1
            ratio_h = box_height / h_img

            if cls == "person":
                count_personas += 1
                if D is not None:
                    distancias_personas.append(D)
                    if self._beep_progresivo(D, "person"):
                        trigger_beep = True
                if (D is not None and D < 2.0) or ratio_w > 0.45 or ratio_h > 0.60:
                    alarma = True
            else:
                count_vehiculos += 1
                if D is not None:
                    distancias_vehiculos.append(D)
                    if self._beep_progresivo(D, "vehiculo"):
                        trigger_beep = True
                if (D is not None and D < 3.0) or ratio_h > 0.60:
                    alarma = True

            color = (0, 255, 0)
            if D is not None:
                if (cls == "person" and D < 2) or (cls != "person" and D < 3):
                    color = (0, 0, 255)
                elif (cls == "person" and D < 5) or (cls != "person" and D < 6):
                    color = (0, 165, 255)
                elif D < 10:
                    color = (0, 255, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls} {conf:.2f} | {D if D is not None else 'NA'}m | {zona}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            resumen.append(label)

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

        try:
            while not self.summary_queue.empty():
                self.summary_queue.get_nowait()
            self.summary_queue.put({"text": "\n".join(resumen), "beep": trigger_beep})
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")
col1, col2 = st.columns([2, 1])
with col1:
    webrtc_ctx = webrtc_streamer(
        key="proximidad-webrtc",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        async_transform=True,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30, "max": 30},
                "facingMode": "environment"
            },
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        video_html_attrs={
            "autoPlay": True,
            "muted": True,
            "playsinline": True,
            "controls": False
        },
    )

with col2:
    st.subheader("Resumen en tiempo real")
    summary_box = st.empty()
    beep_box = st.empty()
    st.caption("Permite acceso a la cámara. Usa HTTPS y un navegador moderno (Chrome/Edge/Firefox).")

if webrtc_ctx and webrtc_ctx.state.playing and webrtc_ctx.video_processor:
    vp = webrtc_ctx.video_processor
    while True:
        try:
            data = vp.summary_queue.get(timeout=0.2)
        except queue.Empty:
            break
        summary_box.text(data["text"])
        if data.get("beep"):
            beep_box.audio(BEEP_WAV, format="audio/wav", start_time=0)
else:
    st.warning("Esperando cámara… Asegúrate de permitir permisos y que el sitio esté en HTTPS.")

