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
    page_icon="",
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
# 2. FUNGSI LOAD DATA & MODEL
# ==========================================
@st.cache_resource
def load_my_model():
    if not os.path.exists('model_starlight.keras'):
        st.error("❌ File 'model_starlight.keras' gak ketemu!")
        return None
    return tf.keras.models.load_model('model_starlight.keras')

@st.cache_data
def load_test_dataset():
    target_file = None
    for root, dirs, files in os.walk("."):
        for file in files:
            if "StarLightCurves_TEST" in file and file.endswith('.arff'):
                target_file = os.path.join(root, file)
                break
    
    if not target_file:
        st.error("❌ File Dataset TEST tidak ditemukan!")
        return None, None
        
    data, meta = arff.loadarff(target_file)
    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    if isinstance(y[0], bytes):
        y = y.astype(str).astype(int)
    else:
        y = y.astype(int)
        
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y

model = load_my_model()
X_test, y_test = load_test_dataset()

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
            st.session_state['idx'] = np.random.randint(0, total_data)
        
        # Ambil dari memori session
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
    true_label_raw = y_test[idx]
    true_label_idx = true_label_raw - 1 if true_label_raw > 0 else true_label_raw

    # --- TAMPILAN GRAFIK ---
    st.subheader(f"📡 Sinyal Cahaya (Data #{idx})")
    
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(sample_signal, color='#00ffcc', linewidth=1.5)
    ax.set_title(f"Visualisasi Kurva Cahaya - Data Nomor {idx}", color='white')
    ax.grid(True, alpha=0.2)
    
    # Styling Gelap
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
        input_data = np.expand_dims(sample_signal, axis=0)
        
        with st.spinner("Sedang menghitung..."):
            probs = model.predict(input_data)
        
        pred_idx = np.argmax(probs)
        confidence = np.max(probs) * 100
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            st.warning(f"**Kunci Jawaban Asli:**\n\n### {CLASSES.get(true_label_idx, 'Unknown')}")
        with c2:
            if pred_idx == true_label_idx:
                st.success(f"**Tebakan AI:**\n\n### {CLASSES[pred_idx]}")
            else:
                st.error(f"**Tebakan AI:**\n\n### {CLASSES[pred_idx]}")
                st.caption("AI Salah Tebak 😅")
        
        st.write(f"Tingkat Keyakinan: **{confidence:.2f}%**")
        st.progress(int(confidence))
        
        with st.expander("Lihat Detail Probabilitas"):
            df_probs = pd.DataFrame(probs, columns=CLASSES.values())
            st.bar_chart(df_probs.T)

else:
    st.warning("Sedang memuat data... Tunggu sebentar.")