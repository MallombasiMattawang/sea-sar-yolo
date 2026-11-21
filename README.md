# 🌊 Sea SAR YOLO — Drone Surveillance

Sistem deteksi **Search and Rescue (SAR) di laut** menggunakan **YOLO11n** yang di-fine-tune pada dataset **SeaDronesSee**. Memproses video drone untuk mengidentifikasi korban/orang di laut, kapal, jetski, dan pelampung.

**Berjalan 100% offline** — tanpa cloud, tanpa database. Output berupa video ber-anotasi, snapshot, dan narasi teks taktis SAR.

---

## ✨ Fitur

- 🎯 Deteksi **swimmer (orang di laut)**, **orang di pantai**, **kapal**, **jetski**, **pelampung/buoy**
- 🤖 Model **YOLO11n SeaDronesSee** (mAP@50: 69.93%) — khusus SAR laut
- 🚨 Risk level otomatis: `NORMAL → MONITORING → WASPADA → DARURAT`
- 📸 Snapshot otomatis dengan **batas maksimal** (`MAX_SNAPSHOTS` — anti-spam)
- 📝 Narasi teks **offline** (template taktis SAR)
- 💾 Output video ber-anotasi dengan **HUD besar + panel narasi**
- ⚡ Optimasi: skip frame, resize output, adaptive tracker
- 🖥️ Jalan langsung di **Windows, macOS, Linux**

---

## 📋 Prasyarat

- **Python 3.9 – 3.11** (disarankan 3.10)
- **pip** & **virtualenv**
- RAM ≥ 4 GB (8 GB disarankan)
- GPU opsional (CUDA / Apple MPS) — bisa jalan di CPU
- Internet (hanya sekali, untuk download model ~6 MB)

Cek versi Python:

```bash
python --version     # Windows
python3 --version    # Linux / macOS
```

---

## 🚀 Instalasi

Pilih sesuai sistem operasi kamu. Semua langkah sudah diuji di **Windows 10/11**, **Ubuntu 22.04**, dan **macOS** (Intel & Apple Silicon).

> ⚠️ **Penting:** Python harus sudah terinstall. Jika belum, download dari https://www.python.org/downloads/ (centang **Add Python to PATH** saat install di Windows).

---

### 🪟 Windows (10 / 11)

#### 1. Clone repository

Buka **PowerShell** atau **CMD**:

```powershell
git clone https://github.com/MallombasiMattawang/sea-sar-yolo.git
cd sea-sar-yolo
```

> Belum punya `git`? Download dari https://git-scm.com/download/win, atau download repo sebagai ZIP dari GitHub lalu extract.

#### 2. Buat virtual environment

**PowerShell:**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

**CMD:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

> **Jika muncul error di PowerShell:**
> ```
> cannot be loaded because running scripts is disabled on this system
> ```
> Jalankan dulu (sekali saja):
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> Lalu ulangi `venv\Scripts\Activate.ps1`.

Setelah aktif, prompt akan muncul `(venv)` di depan:
```
(venv) PS C:\Users\NamaKamu\sea-sar-yolo>
```

#### 3. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Model YOLO11n SeaDronesSee

Model akan **otomatis di-download** dari Hugging Face saat pertama `python main.py` dijalankan. Atau download manual sekarang:

```powershell
python -c "from huggingface_hub import hf_hub_download; import os; p=hf_hub_download(repo_id='dronefreak/seadronessee-yolo11n', filename='best.pt', local_dir='models/sea'); os.rename(p, 'models/sea/seadronessee-yolo11n.pt'); print('OK')"
```

#### 5. Taruh video & jalankan

```powershell
# Copy video ke folder input
copy C:\path\video_kamu.mp4 video-input\

# Jalankan
python main.py
```

---

### 🐧 Linux (Ubuntu / Debian / Fedora)

#### 1. Install prasyarat sistem

**Ubuntu / Debian:**
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git ffmpeg
```

**Fedora:**
```bash
sudo dnf install -y python3 python3-pip git ffmpeg
```

> `python3-venv` wajib — kalau tidak ada, `python3 -m venv` akan error.

#### 2. Clone repository

```bash
git clone https://github.com/MallombasiMattawang/sea-sar-yolo.git
cd sea-sar-yolo
```

#### 3. Buat virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

Setelah aktif, prompt akan muncul `(venv)`:
```
(venv) user@linux:~/sea-sar-yolo$
```

#### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Model YOLO11n SeaDronesSee

Model akan otomatis di-download saat pertama `python main.py`. Atau manual:

```bash
python -c "from huggingface_hub import hf_hub_download; import os; p=hf_hub_download(repo_id='dronefreak/seadronessee-yolo11n', filename='best.pt', local_dir='models/sea'); os.rename(p, 'models/sea/seadronessee-yolo11n.pt'); print('OK')"
```

#### 6. Taruh video & jalankan

```bash
cp ~/Downloads/video_kamu.mp4 video-input/
python main.py
```

#### 7. Server tanpa display (SSH / WSL / headless)

Preview window butuh GUI. Kalau tidak ada display, edit `main.py`:

```python
SHOW_WINDOW = False
```

Program tetap jalan, video output & snapshot tetap tersimpan.

---

### 🍎 macOS (Intel & Apple Silicon M1/M2/M3)

#### 1. Install prasyarat sistem

**Via Homebrew** (disarankan):

```bash
# Install Homebrew dulu kalau belum ada
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python + git + ffmpeg
brew install python@3.10 git ffmpeg
```

**Tanpa Homebrew:** download Python dari https://www.python.org/downloads/macos/ — pilih versi 3.10 atau 3.11.

#### 2. Clone repository

```bash
git clone https://github.com/MallombasiMattawang/sea-sar-yolo.git
cd sea-sar-yolo
```

#### 3. Buat virtual environment

> ⚠️ **Penting untuk macOS:** gunakan `python3`, bukan `python`.

```bash
python3 -m venv venv
source venv/bin/activate
```

Setelah aktif, prompt:
```
(venv) user@MacBook sea-sar-yolo %
```

> **Jika muncul error `zsh: command not found: venvScriptsActivate.ps1`** — itu karena kamu tidak sengaja menyalin perintah Windows. Di macOS perintahnya **`source venv/bin/activate`**, bukan `venv\Scripts\Activate.ps1`.

#### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Catatan Apple Silicon (M1/M2/M3):** `torch` (dependency ultralytics) akan otomatis install versi ARM64. Kalau ada error `numpy`, coba:
> ```bash
> pip install --force-reinstall numpy==1.26.4
> ```

#### 5. Model YOLO11n SeaDronesSee

Model akan otomatis di-download saat pertama `python main.py`. Atau manual:

```bash
python -c "from huggingface_hub import hf_hub_download; import os; p=hf_hub_download(repo_id='dronefreak/seadronessee-yolo11n', filename='best.pt', local_dir='models/sea'); os.rename(p, 'models/sea/seadronessee-yolo11n.pt'); print('OK')"
```

#### 6. Taruh video & jalankan

```bash
cp ~/Downloads/video_kamu.mp4 video-input/
python main.py
```

#### 7. (Opsional) Aktifkan GPU Apple (MPS)

Mac M1/M2/M3 punya GPU terintegrasi yang didukung PyTorch lewat **MPS**. Untuk 2-3x lebih cepat, edit `main.py`:

```python
DEVICE = "mps"       # dari default "cpu"
```

Kalau muncul error `MPS backend out of memory` atau `MPS unavailable`, balik ke `"cpu"`.

---

## ▶️ Cara Menjalankan

Setelah instalasi selesai, cara menjalankan **sama di semua OS**.

### 1. Pastikan venv aktif

Setiap kali buka terminal **baru**, aktifkan venv dulu:

| OS | Perintah aktivasi |
|----|-------------------|
| Windows PowerShell | `venv\Scripts\Activate.ps1` |
| Windows CMD | `venv\Scripts\activate.bat` |
| Linux | `source venv/bin/activate` |
| macOS | `source venv/bin/activate` |

Prompt akan berubah jadi ada `(venv)` di depan. **Kalau tidak ada `(venv)`, berarti belum aktif** dan `python main.py` akan error `ModuleNotFoundError`.

### 2. Taruh video di `video-input/`

```bash
# Windows
copy C:\path\video_kamu.mp4 video-input\

# Linux / macOS
cp ~/Downloads/video_kamu.mp4 video-input/
```

Format didukung: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### 3. Jalankan

```bash
# Proses semua video di folder video-input/
python main.py

# Atau proses 1 video spesifik
python main.py --video video-input/video_kamu.mp4
```

### 4. Output

- **Video ber-anotasi** → `video-output/{nama}_sar_output.mp4`
- **Snapshot** → `snapshots/snap_*.jpg` (maks `MAX_SNAPSHOTS`)
- **Preview live** → jendela OpenCV (tekan **`q`** untuk stop lebih awal)

### 5. Setelah selesai

Buka folder output:

| OS | Perintah |
|----|----------|
| Windows | `start video-output` |
| Linux | `xdg-open video-output` |
| macOS | `open video-output` |

Atau buka manual via File Explorer / Finder.

---

## 🔁 Run Berikutnya (Setelah Setup Awal)

Setup hanya sekali. Untuk run berikutnya:

**Windows PowerShell:**
```powershell
cd C:\path\ke\sea-sar-yolo
venv\Scripts\Activate.ps1
python main.py
```

**Linux:**
```bash
cd ~/path/ke/sea-sar-yolo
source venv/bin/activate
python main.py
```

**macOS:**
```bash
cd ~/path/ke/sea-sar-yolo
source venv/bin/activate
python main.py
```

---

## 📋 Perbandingan Perintah Per OS

| Langkah | Windows PowerShell | Windows CMD | Linux / macOS |
|---------|-------------------|-------------|---------------|
| Buat venv | `python -m venv venv` | `python -m venv venv` | `python3 -m venv venv` |
| Aktifkan venv | `venv\Scripts\Activate.ps1` | `venv\Scripts\activate.bat` | `source venv/bin/activate` |
| Deaktivasi | `deactivate` | `deactivate` | `deactivate` |
| Path python di venv | `venv\Scripts\python.exe` | `venv\Scripts\python.exe` | `venv/bin/python` |
| Copy video | `copy X.mp4 video-input\` | `copy X.mp4 video-input\` | `cp X.mp4 video-input/` |
| Buka folder output | `start video-output` | `start video-output` | `xdg-open` (Linux) / `open` (Mac) |

> 💡 **Tips:** jangan paste baris **komentar** (yang diawali `#`) ke terminal zsh / PowerShell — akan muncul error `command not found` atau `no such user or named directory`. Komentar hanya valid di dalam file `.py`.

---

## 🛠️ Opsi CLI

```bash
# Proses 1 video spesifik
python main.py --video video-input/video.mp4

# Proses semua video di folder video-input/ (default)
python main.py
```

---

## ⚙️ Konfigurasi

Semua pengaturan ada di **blok paling atas `main.py`** (`# ⚙️ KONFIGURASI`). Tinggal edit nilai, tidak perlu sentuh logika.

### Pengaturan Umum

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `VIDEO_INPUT_DIR` | `"video-input"` | Folder input video |
| `VIDEO_OUTPUT_DIR` | `"video-output"` | Folder output video |
| `MODEL_FILENAME` | `"seadronessee-yolo11n.pt"` | Nama file model di `models/sea/` |
| `CONF_THRESHOLD` | `0.25` | Threshold deteksi (turunkan jika objek kecil terlewat) |
| `IMG_SIZE` | `640` | Ukuran inferensi (416/320 lebih cepat) |
| `DEVICE` | `"cpu"` | `"cpu"`, `"mps"` (Mac M-series), `"cuda:0"` (NVIDIA) |

### Optimasi Kecepatan

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `SKIP_FRAMES` | `0` | `0` = semua frame, `1` = proses 1 dari 2 (2x cepat) |
| `RESIZE_OUTPUT` | `True` | Output 1280×720 (file lebih kecil) |
| `USE_HALF_PRECISION` | `False` | FP16 — hanya untuk GPU CUDA |

### Snapshot

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `SAVE_SNAPSHOT` | `True` | Simpan snapshot otomatis |
| `MAX_SNAPSHOTS` | `20` | **Total maksimal snapshot per video** (`0` = unlimited) |
| `SNAPSHOT_DEDUP_SEC` | `8.0` | Jeda detik sebelum objek sama boleh di-snapshot lagi |
| `SNAPSHOT_DIST_TOL` | `100` | Pixel toleransi "objek sama" |

### Tracker (Anti Duplikat)

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `TRACKER_DIST_SWIMMER` | `60` | Jarak toleransi swimmer (px) |
| `TRACKER_DIST_BOAT` | `180` | Jarak toleransi kapal (besar, bergerak jauh) |
| `TRACKER_DIST_JETSKI` | `120` | Jarak toleransi jetski |
| `TRACKER_DIST_LIFESAVING` | `40` | Jarak toleransi pelampung (kecil, diam) |
| `TRACKER_TIMEOUT_SEC` | `8.0` | Objek hilang > 8s dianggap objek baru |

### Tampilan

| Variabel | Default | Keterangan |
|----------|---------|-----------|
| `SHOW_WINDOW` | `True` | Preview live (tekan `q` untuk stop) |
| `SHOW_NARRATION_HUD` | `True` | Tampilkan panel narasi di HUD |
| `HUD_WIDTH` | `420` | Lebar panel HUD |
| `HUD_HEIGHT` | `300` | Tinggi panel HUD |

---

## 🧠 Class yang Dideteksi Model

| ID | Class | Label di HUD | Warna |
|----|-------|--------------|-------|
| 0 | swimmer | `SWIMMER` / `KORBAN/AIR` | Oranye / Merah |
| 1 | boat | `KAPAL` | Kuning |
| 2 | jetski | `JETSKI` | Oranye tua |
| 3 | life_saving_appliances | `PELAMPUNG` | Cyan |
| 4 | buoy | `BUOY` | Hijau muda |

Setiap label menampilkan **confidence score** (misal `KAPAL 0.87`).

> **Korban** ditandai merah hanya jika swimmer **horizontal + ukuran kecil** (indikasi terapung/pingsan). Swimmer yang berenang normal tetap oranye.

---

## 📊 Output

### Video (`video-output/`)

Video ber-anotasi dengan bounding box berwarna sesuai tabel class di atas.

### HUD Kiri Atas

- **Title bar** berwarna sesuai risk level (hijau/kuning/oranye/merah)
- Jumlah korban, orang pantai, kapal, jetski, pelampung
- Total unique (akumulasi sepanjang video)
- Kondisi gelombang & status laut
- Frame counter + snapshot counter (`Snap: 15/20`)

### Panel Narasi (di bawah HUD)

Menampilkan narasi teks taktis terakhir dengan warna header sesuai urgency:
- 🔴 Merah = URGENT (DARURAT)
- 🟡 Kuning = MONITORING
- 🟢 Hijau = NORMAL / CLEAR

### Snapshot (`snapshots/`)

File `.jpg` per trigger objek baru. Nama file: `snap_{tag}_{frame}_{timestamp}.jpg`.

Contoh:
```
snap_normal_30_1730000000123.jpg
snap_darurat_air_120_1730000004567.jpg
snap_monitoring_200_1730000008901.jpg
```

Jumlah maksimal dibatasi `MAX_SNAPSHOTS` (default 20 per video).

---

## 🐛 Troubleshooting

**Q: `ModuleNotFoundError: No module named 'cv2'`**

```bash
pip install -r requirements.txt
```

**Q: `ModuleNotFoundError: No module named 'huggingface_hub'`**

```bash
pip install huggingface_hub
```

**Q: Model gagal download otomatis**

Download manual dari https://huggingface.co/dronefreak/seadronessee-yolo11n, taruh sebagai `models/sea/seadronessee-yolo11n.pt`.

Atau pakai perintah:

```bash
python -c "from huggingface_hub import hf_hub_download; import os; p=hf_hub_download(repo_id='dronefreak/seadronessee-yolo11n', filename='best.pt', local_dir='models/sea'); os.rename(p, 'models/sea/seadronessee-yolo11n.pt'); print('OK')"
```

**Q: Kapal tidak terdeteksi / muncul label `OBJEK`**

Cek terminal saat run — harus muncul:

```
[MODEL] YOLO11n SeaDronesSee: models/sea/seadronessee-yolo11n.pt
[MODEL] Classes: {0: 'swimmer', 1: 'boat', 2: 'jetski', 3: 'life_saving_appliances', 4: 'buoy'}
```

Kalau muncul `[MODEL] Fallback` → model SeaDronesSee belum ada. Download manual.

**Q: Kapal unik terlalu banyak (misal 48 padahal cuma 2)**

Naikkan `TRACKER_DIST_BOAT` di `main.py`:

```python
TRACKER_DIST_BOAT = 250    # dari default 180
```

**Q: Terlalu banyak snapshot**

Turunkan `MAX_SNAPSHOTS`:

```python
MAX_SNAPSHOTS = 10         # dari default 20
```

Atau matikan sekalian:

```python
SAVE_SNAPSHOT = False
```

**Q: Video output corrupt / 0 byte**

Ganti codec di `main.py` (cari `fourcc`):

```python
fourcc = cv2.VideoWriter_fourcc(*"avc1")   # atau "XVID", "MJPG"
```

**Q: Lambat di CPU**

Edit `main.py`:

```python
SKIP_FRAMES = 1            # proses 1 dari 2 (2x cepat)
IMG_SIZE = 416             # dari 640
```

Atau aktifkan GPU:

```python
DEVICE = "mps"             # Mac M1/M2/M3
DEVICE = "cuda:0"          # NVIDIA
```

**Q: Error `MPS backend out of memory` (Mac)**

Kembali ke `DEVICE = "cpu"` atau turunkan `IMG_SIZE = 416`.

**Q: Preview window tidak muncul di WSL / Linux server**

```python
SHOW_WINDOW = False        # jalan headless, output tetap tersimpan
```

**Q: `SyntaxError: unterminated string literal`**

Baris di `main.py` terpotong saat copy-paste. Verifikasi:

```bash
python -m py_compile main.py
```

Kalau error, replace seluruh `main.py` dengan versi dari repo.

**Q: `zsh: command not found: venvScriptsActivate.ps1`**

Kamu menyalin perintah Windows di terminal macOS/Linux. Perintah yang benar:

```bash
source venv/bin/activate
```

**Q: `no such user or named directory`**

Kamu menempel baris komentar shell (`# ...`) di terminal. Komentar hanya valid di file `.py`, bukan di terminal.

---

## 📁 Struktur Project

```
sea-sar-yolo/
├── main.py                       # entry point
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── sea/
│       └── seadronessee-yolo11n.pt    ← auto-download atau manual
├── video-input/                  ← taruh video di sini
│   └── .gitkeep
├── video-output/                 ← hasil video otomatis
│   └── .gitkeep
├── snapshots/                    ← snapshot otomatis
│   └── .gitkeep
└── sample/                       ← video uji (opsional, di-commit)
    └── sample_sea.mp4
```

---

## 🧪 Uji Coba

### Video rekomendasi
- **SeaDronesSee dataset** — video drone SAR laut asli (https://seadronessee.cs.uni-tuebingen.de/)
- **YouTube** — cari "drone sea rescue" / "beach drone footage"
- **Rekam sendiri** — pantai/laut dari drone atau HP

### Pexels / Pixabay (gratis)
- https://www.pexels.com/search/videos/beach%20drone/
- https://pixabay.com/videos/search/sea%20drone/

### Kompres video besar (> 50 MB)

```bash
ffmpeg -i video_asli.mp4 -t 20 -vf "scale=1280:720" -c:v libx264 -crf 30 -an sample/sample_sea.mp4
```

- `-t 20` → 20 detik pertama
- `scale=1280:720` → resolusi HD
- `crf 30` → kompresi (angka besar = kecil)
- `-an` → hapus audio

---

## 📜 Lisensi

MIT License — bebas digunakan untuk penelitian, tugas akhir, dan pengembangan.

---

## 🙏 Kredit

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [SeaDronesSee Dataset](https://seadronessee.cs.uni-tuebingen.de/)
- [YOLO11n SeaDronesSee model (Hugging Face)](https://huggingface.co/dronefreak/seadronessee-yolo11n)
- [DetectionBench](https://github.com/orgs/detectionbench)