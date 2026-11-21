# 🌊 Sea SAR YOLO — Drone Surveillance

Deteksi **Search and Rescue (SAR) di laut** dengan **YOLO11n SeaDronesSee**. Memproses video drone → mengidentifikasi korban, swimmer, kapal, jetski, pelampung. **100% offline**.

---

## ⚡ Quick Start (3 Perintah)

```bash
git clone https://github.com/MallombasiMattawang/sea-sar-yolo.git
cd sea-sar-yolo
python main.py
```

**Semua otomatis** — dependencies, model, dan video sample. Cukup **Python 3.9+**.

Hasil:

- 🎬 Video ber-anotasi → `video-output/`
- 📸 Snapshot → `snapshots/`
- 🖥️ Preview live (tekan `q` untuk stop)

> **Model download:** dari GitHub Release `v1.0` (~5 MB), fallback ke Hugging Face.
> **Pertama kali run:** butuh internet. Setelah itu offline.

---

## 📋 Prasyarat

- **Python 3.9 – 3.11**
- **git** (untuk clone)
- RAM ≥ 4 GB
- Internet (hanya sekali, untuk download dependencies & model)

---

## 🚀 Instalasi Detail (Kalau Quick Start Gagal)

Kalau `python main.py` error, ikuti langkah manual sesuai OS.

### 🪟 Windows

```powershell
# Setup venv
python -m venv venv
venv\Scripts\Activate.ps1

# Kalau error "running scripts is disabled":
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Install deps
pip install -r requirements.txt

# Run
python main.py
```

### 🐧 Linux

```bash
sudo apt install -y python3 python3-pip python3-venv git ffmpeg

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 🍎 macOS

```bash
brew install python@3.10 git ffmpeg

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

> ⚠️ **macOS:** pakai `python3`, bukan `python`. Aktivasi venv = `source venv/bin/activate` (bukan `venv\Scripts\Activate.ps1` yang itu untuk Windows).

---

## 📋 Perbandingan Perintah

| Langkah       | Windows PowerShell          | Windows CMD                 | Linux / macOS                     |
| ------------- | --------------------------- | --------------------------- | --------------------------------- |
| Buat venv     | `python -m venv venv`       | `python -m venv venv`       | `python3 -m venv venv`            |
| Aktifkan venv | `venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` | `source venv/bin/activate`        |
| Copy video    | `copy X.mp4 video-input\`   | `copy X.mp4 video-input\`   | `cp X.mp4 video-input/`           |
| Buka output   | `start video-output`        | `start video-output`        | `open` (Mac) / `xdg-open` (Linux) |

---

## ▶️ Cara Pakai

### 1. Taruh video (opsional)

Video sample sudah ada di `video-input/sample_sea.mp4`. Kalau mau video lain:

```bash
# Windows
copy C:\path\video.mp4 video-input\

# Linux / macOS
cp ~/Downloads/video.mp4 video-input/
```

### 2. Jalankan

```bash
# Proses semua video di video-input/
python main.py

# Atau 1 video spesifik
python main.py --video video-input/video.mp4
```

### 3. Lihat hasil

- Video: `video-output/{nama}_sar_output.mp4`
- Snapshot: `snapshots/snap_*.jpg`

---

## ⚙️ Konfigurasi

Edit blok `# ⚙️ KONFIGURASI` di atas `main.py`. Yang paling sering diubah:

```python
DEVICE = "cpu"          # "mps" (Mac M-series), "cuda:0" (NVIDIA)
IMG_SIZE = 640          # 416/320 = lebih cepat
SKIP_FRAMES = 0         # 1 = proses 1 dari 2 frame (2x cepat)
MAX_SNAPSHOTS = 20      # 0 = unlimited
CONF_THRESHOLD = 0.25   # turunkan kalau objek kecil terlewat
SHOW_WINDOW = True      # False = headless (tanpa preview)
```

### Referensi Lengkap

| Variabel            | Default | Keterangan                                           |
| ------------------- | ------- | ---------------------------------------------------- |
| `CONF_THRESHOLD`    | `0.25`  | Threshold deteksi                                    |
| `IMG_SIZE`          | `640`   | Ukuran inferensi                                     |
| `DEVICE`            | `"cpu"` | `"cpu"` / `"mps"` / `"cuda:0"`                       |
| `SKIP_FRAMES`       | `0`     | Skip frame untuk kecepatan                           |
| `RESIZE_OUTPUT`     | `True`  | Output 1280×720                                      |
| `MAX_SNAPSHOTS`     | `20`    | Batas snapshot per video                             |
| `TRACKER_DIST_BOAT` | `180`   | Toleransi kapal (naikkan kalau kapal dihitung dobel) |
| `SHOW_WINDOW`       | `True`  | Preview live                                         |

---

## 🧠 Class Deteksi

| ID  | Class                  | Label HUD                                          |
| --- | ---------------------- | -------------------------------------------------- |
| 0   | swimmer                | `SWIMMER` / `KORBAN/AIR` (jika horizontal + kecil) |
| 1   | boat                   | `KAPAL`                                            |
| 2   | jetski                 | `JETSKI`                                           |
| 3   | life_saving_appliances | `PELAMPUNG`                                        |
| 4   | buoy                   | `BUOY`                                             |

Setiap label menampilkan confidence (misal `KAPAL 0.87`).

---

## 📊 Output

### Video

Bounding box berwarna + label + confidence.

### HUD Kiri Atas

- Title bar warna sesuai risk (hijau/kuning/oranye/merah)
- Jumlah korban/orang/kapal/jetski/pelampung
- Total unique, kondisi gelombang, frame counter
- Snapshot counter (`Snap: 15/20`)

### Panel Narasi

Narasi taktis terakhir (warna header sesuai urgency):

- 🔴 URGENT = DARURAT
- 🟡 MONITORING
- 🟢 NORMAL

### Snapshot

`snapshots/snap_{tag}_{frame}_{timestamp}.jpg` — maks `MAX_SNAPSHOTS` per video.

---

## 🐛 Troubleshooting

| Error                          | Solusi                                                                                                                                                                       |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: cv2`     | `pip install -r requirements.txt`                                                                                                                                            |
| Model gagal download           | Cek internet, atau download manual dari [Release v1.0](https://github.com/MallombasiMattawang/sea-sar-yolo/releases/tag/v1.0), taruh di `models/sea/seadronessee-yolo11n.pt` |
| Kapal muncul `OBJEK`           | Cek `[MODEL] Classes:` di terminal — harus 5 class. Kalau `[MODEL] Fallback`, model belum ada                                                                                |
| Kapal unik 48 (padahal 2)      | Naikkan `TRACKER_DIST_BOAT = 250`                                                                                                                                            |
| Snapshot terlalu banyak        | Turunkan `MAX_SNAPSHOTS = 10` atau `SAVE_SNAPSHOT = False`                                                                                                                   |
| Video output corrupt           | Ganti codec: `fourcc = cv2.VideoWriter_fourcc(*"avc1")`                                                                                                                      |
| Lambat di CPU                  | `SKIP_FRAMES = 1` + `IMG_SIZE = 416`                                                                                                                                         |
| Mac error `MPS out of memory`  | `DEVICE = "cpu"`                                                                                                                                                             |
| Preview tidak muncul di server | `SHOW_WINDOW = False`                                                                                                                                                        |
| `zsh: command not found`       | Jangan paste perintah Windows (`venv\Scripts\...`) di Mac/Linux. Pakai `source venv/bin/activate`                                                                            |

---

## 📁 Struktur Project

```
sea-sar-yolo/
├── main.py                       # entry point (auto-setup)
├── requirements.txt
├── README.md
├── .gitignore
├── models/sea/                   # model auto-download
├── video-input/                  # taruh video di sini
├── video-output/                 # hasil video
└── snapshots/                    # snapshot
```

---

## 🧪 Video Uji

Sudah tersedia: `video-input/sample_sea.mp4` (37 MB).

Butuh video lain? Cari di:

- Rekam sendiri dari drone/HP

Kompres kalau > 50 MB:

```bash
ffmpeg -i video.mp4 -t 20 -vf "scale=1280:720" -c:v libx264 -crf 30 -an video-input/video.mp4
```

---

## 📜 Lisensi

MIT License.

---

## 🙏 Kredit

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [SeaDronesSee Dataset](https://seadronessee.cs.uni-tuebingen.de/)
- [YOLO11n SeaDronesSee model](https://huggingface.co/dronefreak/seadronessee-yolo11n)
