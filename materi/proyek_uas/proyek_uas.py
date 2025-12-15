import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.io import arff

# ==========================================
# 1. KONFIGURASI TAMPILAN WEB
# ==========================================
st.set_page_config(
    page_title="Proyek UAS - StarLightCurves",
    page_icon="⭐",
    layout="centered"
)

# Header & Judul
st.title("Proyek UAS - StarLightCurves")
st.markdown("""
**Nama :** Ahmad Haydar Al Abror  
**NIM:** 230411100105 

Aplikasi ini mendeteksi jenis bintang (**Cepheid, RR Lyrae, Binary**) berdasarkan pola gelombang cahayanya.
""")
st.markdown("---")

# ==========================================
# 2. FUNGSI LOAD DATA & MODEL (ANTI NYASAR)
# ==========================================
@st.cache_resource
def load_my_model():
    try:
        # GPS: Cari folder tempat file script ini berada
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model_starlight.keras')
        
        # Cek file ada atau tidak
        if not os.path.exists(model_path):
            st.error(f"❌ File model tidak ditemukan di: {model_path}")
            return None
            
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_test_dataset():
    # GPS: Cari file dataset relative terhadap script ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Coba cari di folder 'dataset' (sesuai struktur GitHub kamu)
    target_file = os.path.join(current_dir, 'dataset', 'StarLightCurves_TEST.arff')
    
    # Kalau gak ketemu, coba cari pake os.walk (cadangan)
    if not os.path.exists(target_file):
        for root, dirs, files in os.walk(current_dir):
            for file in files:
                if "StarLightCurves_TEST" in file and file.endswith('.arff'):
                    target_file = os.path.join(root, file)
                    break
    
    if not os.path.exists(target_file):
        st.error("❌ File Dataset TEST tidak ditemukan! Pastikan file .arff sudah di-upload.")
        return None, None
        
    try:
        data, meta = arff.loadarff(target_file)
        df = pd.DataFrame(data)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        # Normalisasi label jika formatnya bytes (b'1' -> 1)
        if isinstance(y[0], bytes):
            y = y.astype(str).astype(int)
        else:
            y = y.astype(int)
            
        # Reshape data untuk model CNN (Sample, TimeSteps, Features)
        X = X.reshape(X.shape[0], X.shape[1], 1)
        return X, y
    except Exception as e:
        st.error(f"Gagal membaca dataset: {e}")
        return None, None

# Load Resources
model = load_my_model()
X_test, y_test = load_test_dataset()

# Mapping Kelas (Sesuaikan dengan model training kamu)
CLASSES = {
    0: "Cepheid (Bintang Berdenyut)", 
    1: "RR Lyrae (Berdenyut Cepat)", 
    2: "Eclipsing Binary (Gerhana)"
}

# ==========================================
# 3. INTERFACE UTAMA
# ==========================================
if model is not None and X_test is not None:
    
    # --- SIDEBAR (PANEL KONTROL) ---
    st.sidebar.header("🎛️ Panel Kontrol")
    
    # PILIHAN METODE INPUT
    input_method = st.sidebar.radio(
        "Pilih Metode Input:",
        ("🎲 Acak (Random)", "🔢 Pilih Manual (Index)")
    )
    
    total_data = len(X_test)
    idx = 0 # Default awal
    
    if input_method == "🎲 Acak (Random)":
        st.sidebar.info(f"Mengambil sampel acak dari {total_data} data.")
        if st.sidebar.button("Kocok Data Baru"):
            # Simpan index acak di session state biar gak berubah pas reload
            st.session_state['idx'] = np.random.randint(0, total_data)
        
        idx = st.session_state.get('idx', 0)

    else: # MODE MANUAL
        st.sidebar.info(f"Masukkan nomor urut data (0 - {total_data-1}).")
        # Input Angka
        user_input = st.sidebar.number_input(
            "Nomor Data:", 
            min_value=0, 
            max_value=total_data-1, 
            value=0,
            step=1
        )
        idx = user_input
        st.session_state['idx'] = idx

    # --- PROSES DATA TERPILIH ---
    sample_signal = X_test[idx]
    
    # Label asli (Handling label dataset yang mungkin mulai dari 1, sedangkan model dari 0)
    true_label_raw = y_test[idx]
    # Asumsi: Jika label dataset 1,2,3 -> ubah jadi 0,1,2
    true_label_idx = true_label_raw - 1 if true_label_raw > 0 else true_label_raw

    # --- TAMPILAN GRAFIK ---
    st.subheader(f"📡 Sinyal Cahaya (Data #{idx})")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sample_signal, color='#00ffcc', linewidth=1.5)
    ax.set_title(f"Visualisasi Kurva Cahaya - Data Nomor {idx}", color='white')
    ax.grid(True, alpha=0.2)
    
    # Styling Gelap (Biar nyatu sama tema Streamlit Dark Mode)
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_color('#0E1117')
    ax.spines['right'].set_color('#0E1117')
    
    st.pyplot(fig)
    
    # --- BAGIAN PREDIKSI ---
    col_btn, col_res = st.columns([1, 2])
    
    with col_btn:
        st.write("") 
        st.write("") 
        btn_predict = st.button("🔍 CEK JENIS BINTANG", type="primary")
        
    if btn_predict:
        # Tambah dimensi batch (1, 1024, 1)
        input_data = np.expand_dims(sample_signal, axis=0)
        
        with st.spinner("Sedang menghitung..."):
            probs = model.predict(input_data)
        
        pred_idx = np.argmax(probs)
        confidence = np.max(probs) * 100
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            label_asli = CLASSES.get(true_label_idx, f"Unknown ({true_label_idx})")
            st.warning(f"**Kunci Jawaban Asli:**\n\n### {label_asli}")
        with c2:
            if pred_idx == true_label_idx:
                st.success(f"**Tebakan AI:**\n\n### {CLASSES[pred_idx]}")
            else:
                st.error(f"**Tebakan AI:**\n\n### {CLASSES[pred_idx]}")
                st.caption("AI Salah Tebak 😅")
        
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
        st.progress(int(confidence))
        
        with st.expander("Lihat Detail Probabilitas"):
            # Buat dataframe probabilitas biar rapi
            df_probs = pd.DataFrame(probs.T, index=CLASSES.values(), columns=["Probabilitas"])
            st.bar_chart(df_probs)

else:
    st.warning("Sedang memuat model dan dataset... Jika pesan ini muncul terus, ada file yang hilang.")