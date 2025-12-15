# ============================================================
# 🎙 Sistem Pengenalan Suara & Kata (Buka / Tutup)
# VERSI LENGKAP DENGAN PERBAIKAN
# ============================================================

# --------- PATCH 1: TORCHAUDIO DUMMY (FIXED) ---------
import types, sys
if "torchaudio" not in sys.modules:
    torchaudio = types.SimpleNamespace()
    torchaudio.list_audio_backends = lambda: ["sox_io"]
    torchaudio.load = lambda *a, **k: (None, None)
    torchaudio.save = lambda *a, **k: None
    # PERBAIKAN 1: Mengembalikan 'info' yang benar
    torchaudio.info = lambda *a, **k: types.SimpleNamespace(sample_rate=16000, num_channels=1) 
    # PERBAIKAN 2: Menggunakan double underscore '__version__'
    torchaudio.__version__ = "2.2.0" 
    sys.modules["torchaudio"] = torchaudio
# -----------------------------------------------------

# --------- PATCH 2: HF_HUB_DOWNLOAD (FIXED) ---------
# Untuk memperbaiki error "unexpected keyword argument 'use_auth_token'"
import huggingface_hub as hf
import inspect

if "use_auth_token" not in inspect.signature(hf.hf_hub_download).parameters:
    original_download = hf.hf_hub_download
    def patched_hf_hub_download(*args, **kwargs):
        # Jika 'use_auth_token' ada, ganti namanya menjadi 'token'
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return original_download(*args, **kwargs)
    hf.hf_hub_download = patched_hf_hub_download
# ----------------------------------------------------

import streamlit as st
import os, numpy as np, random, torch, librosa, librosa.effects, soundfile as sf, re, difflib, tempfile
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import whisper
from speechbrain.inference import SpeakerRecognition
import noisereduce as nr
from pydub import AudioSegment
from streamlit_mic_recorder import mic_recorder

# =================== KONFIGURASI DASAR ===================
BASE_DIR = os.path.join("Data", "data_suara")
EMB_DIR, DATA_DIR = "embeddings", "data"
os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="AI Voice Recognition", page_icon="🎧", layout="wide")

# ====================== CSS ======================
st.markdown("""
<style>
body, [class*="stAppViewContainer"] {
    background-color:#050608;color:#E6E6E6;font-family:'Poppins',sans-serif;
}
.main-title {text-align:center;font-size:2.4rem;font-weight:800;
    background:linear-gradient(90deg,#00FFC6,#00A3FF);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    text-shadow:0 0 25px rgba(0,255,220,.3);
    animation:glow 4s ease-in-out infinite;}
@keyframes glow {0%{text-shadow:0 0 20px rgba(0,255,200,.2);}
    50%{text-shadow:0 0 35px rgba(0,255,255,.5);}
    100%{text-shadow:0 0 20px rgba(0,255,200,.2);}}
.block {background:rgba(255,255,255,.04);
    border:1px solid rgba(255,255,255,.1);
    border-radius:18px;padding:20px 25px;
    margin-bottom:15px;box-shadow:0 10px 40px rgba(0,0,0,.4);}
.stButton>button {
    background:linear-gradient(90deg,#00FFC6,#00A3FF);
    color:#041214;font-weight:700;border:none;border-radius:12px;
    height:48px;width:100%;box-shadow:0 6px 20px rgba(0,255,220,.3);}
.stButton>button:hover {transform:scale(1.04);}
.success-card {background:linear-gradient(90deg,#00FFB3,#00A3FF);color:#041214; padding: 18px; border-radius: 12px;}
.error-card {background:linear-gradient(90deg,#FF4E50,#F9D423);color:#041214; padding: 18px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 class='main-title'>🎙 Sistem Pengenalan Suara & Kata (Buka / Tutup)</h2>", unsafe_allow_html=True)

# ====================== LOAD MODEL ======================
# PERBAIKAN 3: Memastikan tidak ada karakter 'non-printable' (U+00A0)
# Kode ini mengasumsikan Anda sudah mengunduh file model secara manual
# ke folder "pretrained_resnet" untuk menghindari error 404.
with st.spinner("🚀 Memuat model..."):
    recognizer = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-resnet-voxceleb",
        savedir="pretrained_resnet",
        run_opts={"device":"cpu"},
        use_auth_token=None # Aman diatur None karena patch di atas akan menanganinya
    )
    asr_model = whisper.load_model("base")
st.success("✅ Model siap digunakan!", icon="🤖")

# ====================== FUNGSI UTILITAS ======================
def convert_to_wav(input_path, output_path="converted.wav"):
    ext = os.path.splitext(input_path)[-1].lower()
    if ext in [".mp3", ".mp4"]:
        audio = AudioSegment.from_file(input_path)
        audio.export(output_path, format="wav")
        return output_path
    return input_path

def compute_embedding(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=35)
    y = librosa.effects.preemphasis(y)
    y = y / (np.max(np.abs(y)) + 1e-8)
    signal = torch.tensor(y).unsqueeze(0)
    emb = recognizer.encode_batch(signal).squeeze().detach().numpy()
    return emb / (np.linalg.norm(emb)+1e-8)

def recognize_word(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y = nr.reduce_noise(y=y, sr=sr)
    sf.write("clean.wav", y, sr)
    result = asr_model.transcribe("clean.wav", language="id")
    text = re.sub(r"[^a-z\s]", "", result["text"].lower().strip())
    st.write(f"🗣 Transkrip: {text}")
    text = re.sub(r"(.)\\1{2,}", r"\\1", text).strip()
    if "buka" in text: return "buka"
    if "tutup" in text: return "tutup"
    return "tidak dikenali"

def load_all_embeddings():
    embs = {}
    if not os.path.exists(EMB_DIR): return embs
    for u in os.listdir(EMB_DIR):
        path = os.path.join(EMB_DIR, u)
        if os.path.isdir(path):
            embs[u] = {}
            for f in os.listdir(path):
                if f.endswith(".npy"):
                    embs[u][f[:-4]] = np.load(os.path.join(path, f))
    return embs

def plot_similarity(new_emb, embs, word):
    fig, ax = plt.subplots()
    users, sims = [], []
    for u, ws in embs.items():
        if word in ws:
            sim = 1 - cosine(new_emb, ws[word])
            users.append(u); sims.append(sim)
    ax.bar(users, sims, color="#00FFC6"); ax.set_ylim(0,1)
    ax.set_title(f"Perbandingan Speaker ({word})", color="#00FFC6")
    st.pyplot(fig)
    return users, sims

# ====================== TABS ======================
tab_train, tab_register, tab_test = st.tabs(["🧩 Latih Dataset", "🧑‍💻 Daftar Pengguna", "🎧 Uji Identifikasi"])

# ---------- TAB LATIH DATASET ----------
with tab_train:
    st.markdown("<div class='block'><b>🧩 Latih Dataset Otomatis</b></div>", unsafe_allow_html=True)
    if st.button("🚀 Jalankan Pelatihan Dataset"):
        if not os.path.exists(BASE_DIR):
            st.error(f"Folder dataset tidak ditemukan: {BASE_DIR}")
        else:
            persons = [p for p in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR,p))]
            bar = st.progress(0); total = max(1,len(persons)*2); step = 0
            for person in persons:
                for word_folder, word_key in [("Buka","buka"),("Tutup","tutup")]:
                    folder = os.path.join(BASE_DIR,person,word_folder)
                    if not os.path.exists(folder): continue
                    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith((".wav",".mp3",".mp4"))]
                    if not files: continue
                    st.write(f"🎧 {person} - {word_folder}: {len(files)} file")
                    
                    # Pastikan folder embedding user ada
                    user_emb_dir = os.path.join(EMB_DIR, person)
                    os.makedirs(user_emb_dir, exist_ok=True)

                    converted = [convert_to_wav(f, f+".wav") for f in files]
                    embs = [compute_embedding(f) for f in converted]
                    emb_avg = np.mean(embs,axis=0); emb_avg /= (np.linalg.norm(emb_avg)+1e-8)
                    
                    # Simpan embedding ke folder yang benar
                    np.save(os.path.join(user_emb_dir, f"{word_key}.npy"), emb_avg)
                    step += 1; bar.progress(step/total)
            st.success("✅ Pelatihan selesai dan embeddings tersimpan.")
            st.balloons()

# ---------- TAB PENDAFTARAN ----------
with tab_register:
    st.markdown("<div class='block'><b>📥 Pendaftaran Pengguna</b></div>", unsafe_allow_html=True)
    user = st.text_input("Nama pengguna")
    word = st.selectbox("Kata target", ["buka", "tutup"])
    method = st.radio("Metode input:", ["🎙 Rekam langsung", "📂 Unggah file"], horizontal=True, key="reg_method")

    path = None # Inisialisasi path
    if method == "🎙 Rekam langsung":
        audio_bytes = mic_recorder(start_prompt="Mulai Rekam 🎤", stop_prompt="Berhenti ⏹", key="rec1")
        if audio_bytes:
            st.audio(audio_bytes["bytes"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes["bytes"])
                path = tmp.name
    else:
        file = st.file_uploader("Pilih file audio (.wav/.mp3/.mp4)", type=["wav","mp3","mp4"], key="reg_file")
        if file:
            # Simpan file yang diupload ke DATA_DIR
            upload_path = os.path.join(DATA_DIR, file.name)
            with open(upload_path,"wb") as f: f.write(file.read())
            path = convert_to_wav(upload_path) # Konversi jika perlu

    if st.button("🧠 Proses Pendaftaran"):
        if not user.strip() or not word.strip():
            st.warning("Isi semua kolom dulu.")
        elif path is None: # Periksa apakah path sudah di-set
            st.warning("Belum ada audio!")
        else:
            os.makedirs(os.path.join(EMB_DIR,user), exist_ok=True)
            emb = compute_embedding(path)
            np.save(os.path.join(EMB_DIR,user,f"{word}.npy"), emb)
            st.markdown(f"<div class='success-card'>✅ {user} terdaftar untuk kata '{word}'.</div>", unsafe_allow_html=True)
            st.balloons()

# ---------- TAB UJI ----------
with tab_test:
    st.markdown("<div class='block'><b>🎧 Uji Identifikasi</b></div>", unsafe_allow_html=True)
    mode = st.radio("Sumber suara:", ["🎙 Rekam langsung", "📂 Unggah file"], horizontal=True, key="test_method")
    
    test_path = None # Inisialisasi test_path
    if mode == "🎙 Rekam langsung":
        audio_bytes = mic_recorder(start_prompt="Mulai Rekam 🎤", stop_prompt="Berhenti ⏹", key="rec2")
        if audio_bytes:
            st.audio(audio_bytes["bytes"])
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes["bytes"])
                test_path = tmp.name
    else:
        file = st.file_uploader("Unggah file uji (.wav/.mp3/.mp4)", type=["wav","mp3","mp4"], key="test_file")
        if file:
            # Simpan file uji ke DATA_DIR
            upload_path = os.path.join(DATA_DIR, file.name)
            with open(upload_path,"wb") as f: f.write(file.read())
            test_path = convert_to_wav(upload_path) # Konversi jika perlu

    if st.button("▶ Jalankan Uji"):
        if test_path is None: # Periksa apakah test_path sudah di-set
            st.warning("Belum ada file atau rekaman.")
        else:
            st.audio(test_path)
            word = recognize_word(test_path)
            if word in ["buka","tutup"]:
                embs = load_all_embeddings()
                if not embs:
                     st.error("Belum ada pengguna yang terdaftar di database.")
                else:
                    new_emb = compute_embedding(test_path)
                    users, sims = plot_similarity(new_emb, embs, word)
                    if not sims:
                        st.warning(f"Tidak ada data '{word}' terdaftar.")
                    else:
                        best_idx = int(np.argmax(sims))
                        best_user, best_score = users[best_idx], sims[best_idx]
                        
                        # Atur threshold sederhana
                        THRESHOLD = 0.55 # Anda bisa buat ini lebih canggih
                        
                        st.write(f"Skor Tertinggi: {best_user} ({best_score:.3f}) | Threshold: {THRESHOLD}")
                        
                        if best_score >= THRESHOLD:
                            st.markdown(f"<div class='success-card'>✅ Suara dikenali sebagai <b>{best_user}</b> ({word.upper()})</div>", unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f"<div class='error-card'>❌ Akses Ditolak. Suara tidak cocok (skor: {best_score:.3f}).</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='error-card'>❌ Kata tidak dikenali.</div>", unsafe_allow_html=True)