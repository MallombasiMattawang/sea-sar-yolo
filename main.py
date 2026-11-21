"""
Sea SAR YOLO11n - Drone Surveillance (All-in-One, Auto-Setup)
==============================================================
User cukup jalankan: python main.py
- Auto-install dependencies (ultralytics, opencv, huggingface_hub)
- Auto-download model dari GitHub Release / Hugging Face
- Auto-proses semua video di video-input/
"""

import os
import sys
import subprocess
import glob
import time
import argparse

# ==============================================================================
# AUTO-BOOTSTRAP: Install dependencies kalau belum ada
# ==============================================================================
def _ensure_dependencies():
    """Cek & install dependencies yang diperlukan kalau belum terinstall."""
    required = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "ultralytics": "ultralytics",
        "huggingface_hub": "huggingface_hub",
    }
    missing = []
    for module, pkg in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("=" * 60)
        print("  INSTALASI DEPENDENCIES (otomatis)")
        print("=" * 60)
        print(f"  Package belum ada: {', '.join(missing)}")
        print(f"  Menginstall... (butuh 1-3 menit, sekali saja)")
        print("=" * 60)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 "--upgrade"] + missing
            )
            print("[OK] Dependencies terinstall. Melanjutkan...\n")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Gagal install: {e}")
            print(f"  Coba manual: pip install {' '.join(missing)}")
            sys.exit(1)

_ensure_dependencies()

# Import setelah dependencies dipastikan ada
import cv2
import numpy as np
from ultralytics import YOLO


# ==============================================================================
# ⚙️ KONFIGURASI — EDIT DI SINI (opsional)
# ==============================================================================
VIDEO_INPUT_DIR  = "video-input"
VIDEO_OUTPUT_DIR = "video-output"
MODELS_DIR       = "models"
SNAPSHOT_DIR     = "snapshots"

# --- Model & Download URL ---
MODEL_FILENAME = "seadronessee-yolo11n.pt"
MODEL_GITHUB_URL = (
    "https://github.com/MallombasiMattawang/sea-sar-yolo/releases/download/"
    "v1.0/seadronessee-yolo11n.pt"
)
MODEL_HF_REPO = "dronefreak/seadronessee-yolo11n"

CONF_THRESHOLD = 0.25
IMG_SIZE       = 640
DEVICE         = "cpu"      # "mps" untuk Mac M1/M2/M3, "cuda:0" untuk NVIDIA

# --- Optimasi ---
SKIP_FRAMES        = 0
RESIZE_OUTPUT      = True
USE_HALF_PRECISION = False

# --- Garis pantai ---
SHORELINE_RATIO = 0.35

# --- Narasi ---
COOLDOWN_EMERGENCY = 3.0
COOLDOWN_NORMAL    = 5.0

# --- Output ---
SAVE_SNAPSHOT = True
SHOW_WINDOW   = True
OUTPUT_FPS    = None

# --- Snapshot ---
MAX_SNAPSHOTS      = 20
SNAPSHOT_DEDUP_SEC = 8.0
SNAPSHOT_DIST_TOL  = 100

# --- Tracker ---
TRACKER_DIST_SWIMMER    = 60
TRACKER_DIST_ONSHORE    = 60
TRACKER_DIST_BOAT       = 180
TRACKER_DIST_JETSKI     = 120
TRACKER_DIST_LIFESAVING = 40
TRACKER_TIMEOUT_SEC     = 8.0

# --- HUD ---
HUD_WIDTH          = 420
HUD_HEIGHT         = 300
HUD_FONT_SCALE     = 0.55
SHOW_NARRATION_HUD = True
NARRATION_HUD_WIDTH = 420
NARRATION_HUD_LINES = 5
NARRATION_HUD_FONT  = 0.5

# --- Mapping class ---
CLASS_SWIMMER    = 0
CLASS_BOAT       = 1
CLASS_JETSKI     = 2
CLASS_LIFESAVING = 3
CLASS_BUOY       = 4
# ==============================================================================


# ==============================================================================
# NARRATOR
# ==============================================================================
class SeaTelemetryNarrator:
    CLEAR_TEMPLATES = [
        "Sektor pemantauan laut aman. Tidak terdeteksi korban atau objek mencurigakan.",
        "Area laut steril. Belum terlihat aktivitas atau potensi korban.",
        "Pantauan udara menunjukkan area laut clear. Tidak ada objek yang butuh pertolongan.",
    ]
    MONITORING_TEMPLATES = [
        "Terpantau {swimmer} orang di laut dan {onshore} orang di darat. {boat} kapal dan {jetski} jetski beroperasi. Situasi dalam pengawasan.",
        "Aktivitas perairan: {swimmer} orang di laut, {onshore} di darat. {boat} kapal terdeteksi. Kondisi terkendali.",
    ]
    URGENT_TEMPLATES = [
        "PERINGATAN SAR LAUT! Terdeteksi {korban} korban di laut membutuhkan pertolongan! Detail: {detail}.",
        "BAHAYA LAUT! Ditemukan {korban} orang terjebak di perairan ({detail}). Tim SAR diimbau meluncur!",
    ]

    @classmethod
    def generate(cls, metadata):
        risk = metadata.get("risk_level", "NORMAL")
        swimmer = metadata.get("swimmer_count", 0)
        onshore = metadata.get("onshore_count", 0)
        korban = metadata.get("victim_count", 0)
        boat = metadata.get("boat_count", 0)
        jetski = metadata.get("jetski_count", 0)
        lifesaving = metadata.get("lifesaving_count", 0)

        if risk == "NORMAL" or (swimmer == 0 and onshore == 0 and boat == 0):
            return np.random.choice(cls.CLEAR_TEMPLATES), "PATROLI LAUT"

        if risk == "DARURAT AIR" or korban > 0:
            details = []
            if swimmer > 0:
                details.append(f"{swimmer} orang di laut berpotensi korban")
            if lifesaving > 0:
                details.append(f"{lifesaving} pelampung terdeteksi")
            if boat > 0 or jetski > 0:
                vessel = (f"{boat} kapal & {jetski} jetski"
                          if boat and jetski else
                          (f"{boat} kapal" if boat else f"{jetski} jetski"))
                details.append(f"{vessel} di area")
            detail_str = " serta ".join(details) if details else "korban terisolasi di laut"
            return np.random.choice(cls.URGENT_TEMPLATES).format(
                korban=korban, detail=detail_str
            ), "PATROLI LAUT (URGENT)"

        return np.random.choice(cls.MONITORING_TEMPLATES).format(
            swimmer=swimmer, onshore=onshore, boat=boat, jetski=jetski
        ), "PATROLI LAUT (MONITORING)"


# ==============================================================================
# MODEL DOWNLOADER (GitHub Release → Hugging Face fallback)
# ==============================================================================
def _download_model(model_path):
    """Download model dari GitHub Release, fallback ke Hugging Face."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("=" * 60)
    print("  DOWNLOAD MODEL")
    print("=" * 60)

    # Coba GitHub Release dulu
    print(f"[1/2] Mencoba GitHub Release...")
    try:
        import urllib.request
        urllib.request.urlretrieve(MODEL_GITHUB_URL, model_path)
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
            print(f"[OK] Model ter-download dari GitHub: {model_path}")
            print("=" * 60)
            return True
    except Exception as e:
        print(f"[WARN] GitHub Release gagal: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)

    # Fallback ke Hugging Face
    print(f"[2/2] Fallback ke Hugging Face ({MODEL_HF_REPO})...")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=MODEL_HF_REPO,
            filename="best.pt",
            local_dir=os.path.dirname(model_path),
        )
        os.replace(downloaded, model_path)
        print(f"[OK] Model ter-download dari Hugging Face: {model_path}")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"[ERROR] Download gagal total: {e}")
        print(f"  Manual: {MODEL_GITHUB_URL}")
        print(f"  Atau: https://huggingface.co/{MODEL_HF_REPO}")
        print(f"  Taruh di: {model_path}")
        return False


# ==============================================================================
# PROCESSOR
# ==============================================================================
class SeaLiveProcessor:
    def __init__(self):
        self.device = DEVICE
        self.conf_threshold = CONF_THRESHOLD
        self.imgsz = IMG_SIZE
        self.shoreline_ratio = SHORELINE_RATIO

        self.load_model()
        self.reset_state()
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    def load_model(self):
        model_path = os.path.join(MODELS_DIR, "sea", MODEL_FILENAME)

        if os.path.exists(model_path) and os.path.getsize(model_path) > 1_000_000:
            print(f"[MODEL] YOLO11n SeaDronesSee: {model_path}")
            self.model = YOLO(model_path)
        else:
            if not _download_model(model_path):
                sys.exit(1)
            self.model = YOLO(model_path)

        print(f"[MODEL] Classes: {self.model.names}")

    def reset_state(self):
        self.tracked_swimmer = []
        self.tracked_onshore = []
        self.tracked_boat = []
        self.tracked_jetski = []
        self.tracked_lifesaving = []

        self.total_unique_swimmers = 0
        self.total_unique_onshore = 0
        self.total_unique_boats = 0
        self.total_unique_jetskis = 0
        self.total_unique_lifesaving = 0

        self.peak_swimmer = 0
        self.peak_onshore = 0
        self.peak_boat = 0
        self.peak_jetski = 0

        self.total_frames = 0
        self.highest_risk = "NORMAL / AMAN"
        self.last_narration_time = 0.0
        self.snapshot_count = 0
        self._last_narration = ""
        self._last_sender = ""

    def _is_horizontal(self, bbox):
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        return (w / float(h)) >= 1.15

    def _update_tracker(self, current, tracked, max_dist=60):
        now = time.time()
        updated = [c for c in tracked if now - c[2] < TRACKER_TIMEOUT_SEC]
        new_count = 0
        for cx, cy in current:
            is_new = True
            for tx, ty, _ in updated:
                if np.hypot(cx - tx, cy - ty) < max_dist:
                    is_new = False
                    break
            if is_new:
                new_count += 1
                updated.append((cx, cy, now))
        return updated, new_count

    def _wrap_text(self, text, max_width, font, font_scale, thickness):
        words = text.split()
        if not words:
            return []
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _estimate_sea_conditions(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap_var > 300:
                wave, state = "Tinggi (>3m)", "LAUT BERGELOMBANG BESAR"
            elif lap_var > 150:
                wave, state = "Sedang (1-3m)", "LAUT BERGELOMBANG"
            elif lap_var > 60:
                wave, state = "Rendah (0.5-1m)", "LAUT BEROMBAK RENDAH"
            else:
                wave, state = "Tenang (<0.5m)", "LAUT TENANG"
            sat = hsv[:, :, 1].mean() / 255.0
            clarity = "Keruh" if sat > 0.4 else ("Sedang" if sat > 0.2 else "Jernih")
            return wave, clarity, state
        except Exception:
            return "N/A", "N/A", "N/A"

    def _save_snapshot(self, frame, tag="snap"):
        if not SAVE_SNAPSHOT:
            return None
        if MAX_SNAPSHOTS > 0 and self.snapshot_count >= MAX_SNAPSHOTS:
            return None
        try:
            ts = int(time.time())
            ms = int((time.time() - ts) * 1000)
            fn = f"snap_{tag}_{self.total_frames}_{ts}{ms:03d}.jpg"
            path = os.path.join(SNAPSHOT_DIR, fn)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.snapshot_count += 1
            print(f"[SNAPSHOT {self.snapshot_count}/{MAX_SNAPSHOTS or 'inf'}] {fn}")
            return fn
        except Exception as e:
            print(f"[ERROR] Snapshot: {e}")
            return None

    def process_frame(self, frame):
        self.total_frames += 1
        h, w = frame.shape[:2]
        display = frame.copy()
        shoreline_y = int(h * self.shoreline_ratio)

        swimmer_c, onshore_c, boat_c, jetski_c, lifesaving_c = [], [], [], [], []
        victim_boxes = []

        results = self.model.predict(
            source=frame, conf=self.conf_threshold,
            imgsz=self.imgsz, device=self.device, verbose=False,
            half=USE_HALF_PRECISION,
        )

        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            cls_ids = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.int().cpu().numpy()
            confs = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [1.0] * len(cls_ids)

            for i, cls_id in enumerate(cls_ids):
                x1, y1, x2, y2 = xyxy[i]
                conf = confs[i]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cls_id == CLASS_SWIMMER:
                    if cy >= shoreline_y:
                        swimmer_c.append((cx, cy))
                        box_area = (x2 - x1) * (y2 - y1)
                        area_ratio = box_area / (w * h)
                        is_horiz = self._is_horizontal([x1, y1, x2, y2])
                        if is_horiz and area_ratio < 0.01:
                            label, color = f"KORBAN/AIR {conf:.2f}", (0, 0, 255)
                            victim_boxes.append([x1, y1, x2, y2])
                        else:
                            label, color = f"SWIMMER {conf:.2f}", (0, 165, 255)
                    else:
                        onshore_c.append((cx, cy))
                        label, color = f"ORANG PANTAI {conf:.2f}", (0, 255, 0)
                elif cls_id == CLASS_BOAT:
                    boat_c.append((cx, cy))
                    label, color = f"KAPAL {conf:.2f}", (255, 255, 0)
                elif cls_id == CLASS_JETSKI:
                    jetski_c.append((cx, cy))
                    label, color = f"JETSKI {conf:.2f}", (255, 165, 0)
                elif cls_id == CLASS_LIFESAVING:
                    lifesaving_c.append((cx, cy))
                    label, color = f"PELAMPUNG {conf:.2f}", (0, 255, 255)
                elif cls_id == CLASS_BUOY:
                    lifesaving_c.append((cx, cy))
                    label, color = f"BUOY {conf:.2f}", (200, 255, 200)
                else:
                    label, color = f"UNKNOWN({cls_id}) {conf:.2f}", (200, 200, 200)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        self.tracked_swimmer, n_sw = self._update_tracker(swimmer_c, self.tracked_swimmer, TRACKER_DIST_SWIMMER)
        self.tracked_onshore, n_on = self._update_tracker(onshore_c, self.tracked_onshore, TRACKER_DIST_ONSHORE)
        self.tracked_boat, n_bt = self._update_tracker(boat_c, self.tracked_boat, TRACKER_DIST_BOAT)
        self.tracked_jetski, n_jk = self._update_tracker(jetski_c, self.tracked_jetski, TRACKER_DIST_JETSKI)
        self.tracked_lifesaving, n_ls = self._update_tracker(lifesaving_c, self.tracked_lifesaving, TRACKER_DIST_LIFESAVING)

        self.total_unique_swimmers += n_sw
        self.total_unique_onshore += n_on
        self.total_unique_boats += n_bt
        self.total_unique_jetskis += n_jk
        self.total_unique_lifesaving += n_ls

        self.peak_swimmer = max(self.peak_swimmer, len(swimmer_c))
        self.peak_onshore = max(self.peak_onshore, len(onshore_c))
        self.peak_boat = max(self.peak_boat, len(boat_c))
        self.peak_jetski = max(self.peak_jetski, len(jetski_c))

        victim_count = len(victim_boxes)
        if victim_count > 0:
            risk, risk_color = "DARURAT AIR", (0, 0, 255)
            self.highest_risk = "BAHAYA UTAMA"
        elif len(swimmer_c) > 0:
            risk, risk_color = "WASPADA LAUT", (0, 165, 255)
            if self.highest_risk != "BAHAYA UTAMA":
                self.highest_risk = "WASPADA LAUT"
        elif boat_c or jetski_c:
            risk, risk_color = "MONITORING", (0, 255, 255)
            if self.highest_risk not in ["BAHAYA UTAMA", "WASPADA LAUT"]:
                self.highest_risk = "MONITORING"
        else:
            risk, risk_color = "NORMAL", (0, 255, 0)

        wave, clarity, state = self._estimate_sea_conditions(frame)

        snapshot_worthy = False
        reasons = []
        if n_sw > 0: snapshot_worthy = True; reasons.append(f"swimmer_x{n_sw}")
        if n_bt > 0: snapshot_worthy = True; reasons.append(f"boat_x{n_bt}")
        if n_jk > 0: snapshot_worthy = True; reasons.append(f"jetski_x{n_jk}")
        if n_ls > 0: snapshot_worthy = True; reasons.append(f"lifesaving_x{n_ls}")
        if n_on > 0: snapshot_worthy = True; reasons.append(f"onshore_x{n_on}")

        if snapshot_worthy:
            meta = {
                "victim_count": victim_count,
                "swimmer_count": len(swimmer_c),
                "onshore_count": len(onshore_c),
                "boat_count": len(boat_c),
                "jetski_count": len(jetski_c),
                "lifesaving_count": len(lifesaving_c),
                "risk_level": risk,
            }
            narration, sender = SeaTelemetryNarrator.generate(meta)
            tag = risk.lower().replace(" ", "_")
            print(f"[{sender}] {narration}  | {', '.join(reasons)}")
            self._save_snapshot(display, tag=tag)
            self.last_narration_time = time.time()
            self._last_narration = narration
            self._last_sender = sender

        # HUD
        hud_x, hud_y = 15, 15
        hud_w, hud_h = HUD_WIDTH, HUD_HEIGHT

        overlay = display.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, display, 0.35, 0, display)
        cv2.rectangle(display, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 80), 2)

        title_h = 42
        cv2.rectangle(display, (hud_x, hud_y), (hud_x + hud_w, hud_y + title_h), risk_color, -1)
        cv2.putText(display, f"SEA SAR [OFFLINE] - {risk}",
                    (hud_x + 15, hud_y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.line(display, (hud_x + 10, hud_y + 50),
                 (hud_x + hud_w - 10, hud_y + 50), (120, 120, 120), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        fs = HUD_FONT_SCALE
        th = 1
        gap = 24
        x0 = hud_x + 18
        y = hud_y + 76

        cv2.putText(display, f"Korban di Laut   : {victim_count}  (Unik: {self.total_unique_swimmers})",
                    (x0, y), font, fs, (60, 60, 255), th, cv2.LINE_AA); y += gap
        cv2.putText(display, f"Orang di Pantai  : {len(onshore_c)}  (Unik: {self.total_unique_onshore})",
                    (x0, y), font, fs, (80, 220, 80), th, cv2.LINE_AA); y += gap
        cv2.putText(display, f"Kapal / Jetski   : {len(boat_c)} / {len(jetski_c)}",
                    (x0, y), font, fs, (255, 235, 60), th, cv2.LINE_AA); y += gap
        cv2.putText(display, f"Pelampung        : {len(lifesaving_c)}",
                    (x0, y), font, fs, (60, 235, 235), th, cv2.LINE_AA); y += gap

        y += 2
        cv2.line(display, (hud_x + 10, y), (hud_x + hud_w - 10, y), (90, 90, 90), 1)
        y += 22

        cv2.putText(display, f"Gelombang  : {wave}",
                    (x0, y), font, fs, (150, 210, 255), th, cv2.LINE_AA); y += gap
        cv2.putText(display, f"Status Laut: {state}",
                    (x0, y), font, fs, (150, 210, 255), th, cv2.LINE_AA); y += gap
        cv2.putText(display, f"Frame      : {self.total_frames}  |  Snap: {self.snapshot_count}/{MAX_SNAPSHOTS or 'inf'}",
                    (x0, y), font, fs, (200, 200, 200), th, cv2.LINE_AA)

        if SHOW_NARRATION_HUD and self._last_narration:
            nar_x = hud_x
            nar_y = hud_y + hud_h + 15
            nar_w = NARRATION_HUD_WIDTH
            nfont = cv2.FONT_HERSHEY_DUPLEX
            nfs = NARRATION_HUD_FONT
            nth = 1
            pad = 12
            line_h = 20

            max_text_w = nar_w - pad * 2
            wrapped = self._wrap_text(self._last_narration, max_text_w, nfont, nfs, nth)
            wrapped = wrapped[:NARRATION_HUD_LINES]

            header_h = 26
            nar_h = header_h + len(wrapped) * line_h + pad * 2

            overlay2 = display.copy()
            cv2.rectangle(overlay2, (nar_x, nar_y), (nar_x + nar_w, nar_y + nar_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay2, 0.7, display, 0.3, 0, display)
            cv2.rectangle(display, (nar_x, nar_y), (nar_x + nar_w, nar_y + nar_h), (200, 200, 200), 2)

            if "URGENT" in self._last_sender:
                header_color = (0, 0, 255)
            elif "MONITORING" in self._last_sender:
                header_color = (0, 255, 255)
            else:
                header_color = (0, 255, 0)

            cv2.rectangle(display, (nar_x, nar_y), (nar_x + nar_w, nar_y + header_h), header_color, -1)
            cv2.putText(display, f"  {self._last_sender}",
                        (nar_x + 8, nar_y + 19),
                        nfont, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            ty = nar_y + header_h + pad + 14
            for line in wrapped:
                cv2.putText(display, line, (nar_x + pad, ty),
                            nfont, nfs, (230, 230, 230), nth, cv2.LINE_AA)
                ty += line_h

        return display


# ==============================================================================
# MAIN
# ==============================================================================
def find_videos(folder):
    exts = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser(description="Sea SAR YOLO11n - All-in-One")
    parser.add_argument("--video", help="Path video spesifik (opsional)")
    args = parser.parse_args()

    os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    print("=" * 60)
    print("  SEA SAR YOLO11n - DRONE SURVEILLANCE")
    print("=" * 60)
    print(f"  Device       : {DEVICE}")
    print(f"  Image size   : {IMG_SIZE}")
    print(f"  Conf         : {CONF_THRESHOLD}")
    print(f"  Max snapshot : {MAX_SNAPSHOTS}")
    print("=" * 60)

    if args.video:
        if not os.path.exists(args.video):
            print(f"[ERROR] File tidak ditemukan: {args.video}")
            sys.exit(1)
        videos = [args.video]
    else:
        videos = find_videos(VIDEO_INPUT_DIR)

    if not videos:
        print(f"\n[INFO] Tidak ada video di '{VIDEO_INPUT_DIR}/'")
        print(f"  Taruh file .mp4 di folder: {os.path.abspath(VIDEO_INPUT_DIR)}")
        sys.exit(0)

    print(f"[INFO] {len(videos)} video ditemukan, mulai proses...\n")

    processor = SeaLiveProcessor()
    total_start = time.time()
    grand_total = 0

    for idx, vid_path in enumerate(videos, 1):
        base = os.path.splitext(os.path.basename(vid_path))[0]
        out_path = os.path.join(VIDEO_OUTPUT_DIR, f"{base}_sar_output.mp4")

        print("\n" + "-" * 60)
        print(f"[{idx}/{len(videos)}] {vid_path}")
        print(f"          -> {out_path}")
        print("-" * 60)

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"[ERROR] Gagal buka: {vid_path}")
            continue

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w, out_h = (1280, 720) if RESIZE_OUTPUT else (w_in, h_in)
        fps_out = OUTPUT_FPS if OUTPUT_FPS else fps_in

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, (out_w, out_h))
        if not writer.isOpened():
            print(f"[ERROR] Writer gagal: {out_path}")
            cap.release()
            continue

        processor.reset_state()
        t0 = time.time()
        frame_read, frame_written = 0, 0
        last_display = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_read += 1

            if SKIP_FRAMES > 0 and (frame_read % (SKIP_FRAMES + 1)) != 1:
                if last_display is not None:
                    out = cv2.resize(last_display, (out_w, out_h)) if RESIZE_OUTPUT else last_display
                    writer.write(out)
                    frame_written += 1
                continue

            out_frame = processor.process_frame(frame)
            last_display = out_frame

            if RESIZE_OUTPUT:
                out_frame = cv2.resize(out_frame, (out_w, out_h))
            writer.write(out_frame)
            frame_written += 1

            if frame_read % 30 == 0 and total_in > 0:
                pct = frame_read / total_in * 100
                el = time.time() - t0
                print(f"  ... {frame_read}/{total_in} ({pct:.1f}%) | {frame_read/max(el,0.001):.1f} fps", end="\r")

            if SHOW_WINDOW:
                cv2.imshow("Sea SAR - YOLO11n", out_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Dihentikan user (q).")
                    break

        cap.release()
        writer.release()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()

        elapsed = time.time() - t0
        grand_total += frame_written

        print(f"\n[SELESAI] {base}")
        print(f"  Frame baca/tulis   : {frame_read} / {frame_written}")
        print(f"  Waktu              : {elapsed:.1f}s ({frame_read/max(elapsed,0.001):.1f} fps)")
        print(f"  Korban unik        : {processor.total_unique_swimmers}")
        print(f"  Orang pantai unik  : {processor.total_unique_onshore}")
        print(f"  Kapal unik         : {processor.total_unique_boats}")
        print(f"  Jetski unik        : {processor.total_unique_jetskis}")
        print(f"  Snapshot tersimpan : {processor.snapshot_count}")
        print(f"  Risiko tertinggi   : {processor.highest_risk}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  Total: {len(videos)} video | {grand_total} frame | {total_elapsed:.1f}s")
    print(f"  Output: {os.path.abspath(VIDEO_OUTPUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()