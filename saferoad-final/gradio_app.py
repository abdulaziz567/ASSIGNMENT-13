"""
gradio_app.py
─────────────
SafeRoad AI — Seatbelt Violation Detection System
Gradio Interface (Assignment Requirement)

Tabs:
  1. Image Detection
  2. Video Detection
  3. Violation Log
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import math
import io
import os

# ─── Tesseract ───────────────────────────────────────────────────────────────
try:
    import pytesseract
    import platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# ─── Real Detector ───────────────────────────────────────────────────────────
try:
    from models.detector import SeatbeltDetector, annotate
    detector = SeatbeltDetector(
        seatbelt_model_path="models/seatbelt_yolov8.pt",
        plate_model_path="models/license_plate_yolov8.pt",
    )
    DETECTOR_AVAILABLE = True
except Exception:
    DETECTOR_AVAILABLE = False
    detector = None

# ─── Database ─────────────────────────────────────────────────────────────────
try:
    from db.database import ViolationDB
    DB_AVAILABLE = True
except Exception:
    DB_AVAILABLE = False

# ─── Session violation log ────────────────────────────────────────────────────
violation_log = []

# ─── Demo detection (when no model) ──────────────────────────────────────────
def _demo_plate():
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"
    return (
        "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
        + "-"
        + "".join(random.choices("0123456789", k=2))
        + "-"
        + "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
    )

def _demo_detect(image_rgb):
    h, w = image_rgb.shape[:2]
    rng = random.Random(int(image_rgb[::30, ::30].mean() * 100))
    n = rng.randint(1, 3)
    dets = []
    for _ in range(n):
        x1 = rng.randint(int(w * 0.05), int(w * 0.5))
        y1 = rng.randint(int(h * 0.05), int(h * 0.4))
        x2 = min(x1 + rng.randint(int(w * 0.2), int(w * 0.45)), w - 1)
        y2 = min(y1 + rng.randint(int(h * 0.3), int(h * 0.6)), h - 1)
        has_belt = rng.random() > 0.5
        conf = round(rng.uniform(0.60, 0.97), 2)
        plate = None if has_belt else _demo_plate()
        dets.append({
            "bbox": (x1, y1, x2, y2),
            "has_belt": has_belt,
            "confidence": conf,
            "plate": plate,
            "plate_bbox": None,
        })
    return dets

# ─── Annotate image ───────────────────────────────────────────────────────────
def _annotate(image_rgb, detections):
    img = image_rgb.copy()
    COLOR_SAFE = (0, 200, 100)
    COLOR_VIOL = (220, 50, 50)
    COLOR_PLATE = (255, 165, 0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        has_belt = det.get("has_belt", True)
        conf = det.get("confidence", 0.0)
        plate = det.get("plate")
        color = COLOR_SAFE if has_belt else COLOR_VIOL
        label = f"✓ Belt  {conf:.0%}" if has_belt else f"✗ VIOLATION  {conf:.0%}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 12), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if plate and not has_belt:
            plate_label = f"Plate: {plate}"
            (pw, ph), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y2 + 2), (x1 + pw + 10, y2 + ph + 14), COLOR_PLATE, -1)
            cv2.putText(img, plate_label, (x1 + 5, y2 + ph + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Image Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_image(image, conf_threshold, db_url):
    global violation_log

    if image is None:
        return None, "⚠️ Please upload an image.", ""

    image_rgb = np.array(image)

    # Run detection
    if DETECTOR_AVAILABLE and detector:
        detector.conf_threshold = conf_threshold
        detections = detector.detect(image_rgb)
    else:
        detections = _demo_detect(image_rgb)

    # Annotate
    annotated = _annotate(image_rgb, detections)

    # Build result text
    violations = [d for d in detections if not d["has_belt"]]
    safe       = [d for d in detections if d["has_belt"]]

    result_text = f"### 🔍 Detection Results\n\n"
    result_text += f"- **Total Detected:** {len(detections)}\n"
    result_text += f"- **✅ Safe (with belt):** {len(safe)}\n"
    result_text += f"- **❌ Violations:** {len(violations)}\n\n"

    # Store violations
    for det in violations:
        plate = det.get("plate") or "UNKNOWN"
        conf  = det.get("confidence", 0.0)
        entry = {
            "Plate Number": plate,
            "Violation Type": "No Seatbelt",
            "Confidence": f"{conf:.2f}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Source": "Image",
        }
        violation_log.append(entry)

        result_text += f"🚨 **VIOLATION DETECTED**\n"
        result_text += f"- Plate: `{plate}`\n"
        result_text += f"- Confidence: `{conf:.2%}`\n\n"

        # Save to DB
        if db_url and DB_AVAILABLE:
            try:
                db = ViolationDB(db_url)
                db.insert_violation(
                    plate_number=plate,
                    violation_type="No Seatbelt",
                    confidence=conf,
                    image_name="gradio_upload",
                )
                db.close()
                result_text += "✅ Saved to PostgreSQL\n"
            except Exception as e:
                result_text += f"⚠️ DB Error: {e}\n"

    if not violations:
        result_text += "✅ **No violations detected!**\n"

    mode = "🤖 Real YOLO Model" if DETECTOR_AVAILABLE else "🎭 Demo Mode"
    return Image.fromarray(annotated), result_text, mode


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Video Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_video(video_path, conf_threshold, db_url, max_frames):
    global violation_log

    if video_path is None:
        return None, "⚠️ Please upload a video."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "❌ Could not open video."

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video
    out_path = "/tmp/saferoad_output.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    out      = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_num   = 0
    violations  = 0
    processed   = 0
    result_text = ""

    while cap.isOpened() and processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % 3 != 0:  # process every 3rd frame for speed
            out.write(frame)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if DETECTOR_AVAILABLE and detector:
            detector.conf_threshold = conf_threshold
            dets = detector.detect(rgb)
        else:
            dets = _demo_detect(rgb)

        annotated_rgb = _annotate(rgb, dets)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
        out.write(annotated_bgr)

        for det in dets:
            if not det["has_belt"]:
                violations += 1
                plate = det.get("plate") or "UNKNOWN"
                conf  = det.get("confidence", 0.0)
                violation_log.append({
                    "Plate Number":   plate,
                    "Violation Type": "No Seatbelt",
                    "Confidence":     f"{conf:.2f}",
                    "Timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Source":         f"Video Frame {frame_num}",
                })

        processed += 1

    cap.release()
    out.release()

    result_text = f"### 📹 Video Detection Results\n\n"
    result_text += f"- **Frames Processed:** {processed}\n"
    result_text += f"- **Violations Found:** {violations}\n"
    mode = "🤖 Real YOLO" if DETECTOR_AVAILABLE else "🎭 Demo Mode"
    result_text += f"- **Mode:** {mode}\n"

    return out_path, result_text


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Violation Log
# ─────────────────────────────────────────────────────────────────────────────

def get_violation_log(db_url):
    rows = []

    # From DB
    if db_url and DB_AVAILABLE:
        try:
            db = ViolationDB(db_url)
            df = db.fetch_violations()
            db.close()
            if not df.empty:
                return df, f"✅ Loaded {len(df)} records from PostgreSQL"
        except Exception as e:
            pass

    # From session
    if violation_log:
        df = pd.DataFrame(violation_log)
        return df, f"📋 {len(df)} violations in session memory"

    return pd.DataFrame(columns=["Plate Number", "Violation Type",
                                  "Confidence", "Timestamp", "Source"]), \
           "ℹ️ No violations recorded yet."


def export_csv(db_url):
    df, _ = get_violation_log(db_url)
    if df.empty:
        return None
    path = "/tmp/violations_export.csv"
    df.to_csv(path, index=False)
    return path


def clear_log():
    global violation_log
    violation_log = []
    return pd.DataFrame(columns=["Plate Number", "Violation Type",
                                   "Confidence", "Timestamp", "Source"]), \
           "🗑️ Session log cleared."


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="orange",
    secondary_hue="red",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Outfit"),
).set(
    button_primary_background_fill="linear-gradient(135deg, #ff6b00, #ff9a3c)",
    button_primary_text_color="white",
    button_primary_shadow="0 4px 15px rgba(255,107,0,0.4)",
)

with gr.Blocks(
    theme=THEME,
    title="🚦 SafeRoad AI",
    css="""
    .header-box {
        background: linear-gradient(135deg, #1a1a1a, #2d2d2d);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 20px;
        border-left: 5px solid #ff6b00;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #ff6b00;
        margin: 0;
    }
    .header-sub {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 4px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .status-box {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px 16px;
        font-family: monospace;
        font-size: 0.85rem;
    }
    """
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="header-box">
        <div class="header-title">🚦 SafeRoad AI</div>
        <div class="header-sub">Seatbelt Violation Detection & License Plate Logging System</div>
        <div style="margin-top:12px;font-size:0.8rem;color:#888">
            YOLOv8 &nbsp;·&nbsp; Tesseract OCR &nbsp;·&nbsp; PostgreSQL &nbsp;·&nbsp; Gradio
        </div>
    </div>
    """)

    # ── Model status ──────────────────────────────────────────────────────────
    status_color = "#18a96a" if DETECTOR_AVAILABLE else "#ff6b00"
    status_text  = "✅ Real YOLOv8 models loaded — Full pipeline active" \
                   if DETECTOR_AVAILABLE else \
                   "⚠️ Demo mode — Place seatbelt_yolov8.pt & license_plate_yolov8.pt in models/ folder"

    gr.HTML(f"""
    <div class="status-box" style="border-left: 4px solid {status_color}; margin-bottom:16px">
        {status_text}
    </div>
    """)

    # ── Shared inputs ─────────────────────────────────────────────────────────
    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            conf_slider = gr.Slider(
                minimum=0.30, maximum=1.0, value=0.60, step=0.05,
                label="Confidence Threshold"
            )
            db_url_box = gr.Textbox(
                label="PostgreSQL URL (optional)",
                placeholder="postgresql://user:pass@host:5432/dbname",
                type="password",
            )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Tab 1: Image Detection ─────────────────────────────────────────
        with gr.Tab("📷 Image Detection"):
            gr.Markdown("### Upload an image to detect seatbelt violations")
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(
                        type="pil",
                        label="Input Image",
                        height=350,
                    )
                    detect_btn = gr.Button(
                        "🔍 Run Detection",
                        variant="primary",
                        size="lg",
                    )
                with gr.Column():
                    img_output = gr.Image(
                        label="Annotated Output",
                        height=350,
                    )
                    result_md  = gr.Markdown(label="Detection Results")
                    mode_label = gr.Textbox(label="Mode", interactive=False)

            detect_btn.click(
                fn=detect_image,
                inputs=[img_input, conf_slider, db_url_box],
                outputs=[img_output, result_md, mode_label],
            )

        # ── Tab 2: Video Detection ─────────────────────────────────────────
        with gr.Tab("🎬 Video Detection"):
            gr.Markdown("### Upload a video for frame-by-frame seatbelt detection")
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Input Video")
                    max_frames = gr.Slider(
                        minimum=10, maximum=200, value=50, step=10,
                        label="Max Frames to Process"
                    )
                    vid_btn = gr.Button("▶️ Process Video", variant="primary", size="lg")
                with gr.Column():
                    vid_output  = gr.Video(label="Annotated Output")
                    vid_result  = gr.Markdown(label="Results")

            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, conf_slider, db_url_box, max_frames],
                outputs=[vid_output, vid_result],
            )

        # ── Tab 3: Violation Log ───────────────────────────────────────────
        with gr.Tab("🗃️ Violation Log"):
            gr.Markdown("### All detected violations — from database or session memory")
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Log", variant="primary")
                export_btn  = gr.Button("📥 Export CSV", variant="secondary")
                clear_btn   = gr.Button("🗑️ Clear Session", variant="stop")

            log_status = gr.Markdown()
            log_table  = gr.Dataframe(
                headers=["Plate Number", "Violation Type",
                         "Confidence", "Timestamp", "Source"],
                label="Violations",
                interactive=False,
            )
            csv_file = gr.File(label="Download CSV")

            refresh_btn.click(
                fn=get_violation_log,
                inputs=[db_url_box],
                outputs=[log_table, log_status],
            )
            export_btn.click(
                fn=export_csv,
                inputs=[db_url_box],
                outputs=[csv_file],
            )
            clear_btn.click(
                fn=clear_log,
                inputs=[],
                outputs=[log_table, log_status],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center;margin-top:24px;padding:16px;
                border-top:1px solid #333;color:#666;font-size:0.8rem">
        🚦 SafeRoad AI &nbsp;|&nbsp; YOLOv8 + Tesseract OCR + PostgreSQL + Gradio
        &nbsp;|&nbsp;
        <a href="https://huggingface.co/spaces/abdul-aziz-ai/saferoad-ai"
           style="color:#ff6b00" target="_blank">HuggingFace Demo</a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=False)
