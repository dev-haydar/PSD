import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Proyek UAS - StarLightCurves",
    page_icon="⭐"
)

# --- 2. JUDUL & IDENTITAS (Sesuai Screenshot) ---
st.title("Proyek UAS - StarLightCurves")
st.markdown("**Nama :** Ahmad Haydar Al Abror")
st.markdown("**NIM:** 230411100105")
st.write("Aplikasi ini mendeteksi jenis bintang (**Cepheid, RR Lyrae, Binary**) berdasarkan pola gelombang cahayanya.")
st.markdown("---")

# --- 3. LOAD MODEL (BAGIAN KRUSIAL FIX PATH) ---
@st.cache_resource
def load_model():
    try:
        # Cari tahu folder tempat file script ini (proyek_uas.py) berada
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Gabungkan folder tersebut dengan nama file model
        model_path = os.path.join(current_dir, 'model_starlight.keras')
        
        # Load modelnya
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error: Gagal memuat model!\nPath: {model_path}\nPesan Error: {e}")
        return None

# Panggil fungsi load model
model = load_model()

# --- 4. INTERFACE PENGGUNA (INPUT DATA) ---
if model is not None:
    st.success("✅ Model berhasil dimuat! Sistem siap digunakan.")
    
    # Area untuk upload file (Sesuaikan dengan cara kamu input data)
    st.subheader("Input Data Gelombang Cahaya")
    uploaded_file = st.file_uploader("Upload file (CSV/TXT)", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            # Baca data (Sesuaikan delimiter jika perlu, misal sep=',')
            # Ini contoh standar untuk membaca file CSV tanpa header
            data = pd.read_csv(uploaded_file, header=None)
            
            st.write("Preview Data:")
            st.dataframe(data.head())

            if st.button("🔍 Deteksi Jenis Bintang"):
                # --- PROSES PREDIKSI ---
                # Ubah data menjadi array numpy
                input_data = data.values
                
                # PENTING: Pastikan bentuk input_data sesuai dengan input shape model kamu
                # Jika model butuh 3D (samples, timesteps, features), lakukan reshape di sini
                # Contoh: input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1)
                
                # Lakukan prediksi
                prediction = model.predict(input_data)
                predicted_class = np.argmax(prediction, axis=1)
                
                # Mapping hasil prediksi (Sesuaikan urutan kelas model kamu)
                # Contoh urutan: 0=Cepheid, 1=RR Lyrae, 2=Binary
                label_map = {0: "Cepheid", 1: "RR Lyrae", 2: "Binary"}
                
                # Tampilkan hasil
                st.subheader("Hasil Klasifikasi:")
                for i, kelas in enumerate(predicted_class):
                    label = label_map.get(kelas, "Unknown")
                    confidence = np.max(prediction[i]) * 100
                    st.success(f"Data ke-{i+1}: Jenis **{label}** (Akurasi: {confidence:.2f}%)")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data: {e}")