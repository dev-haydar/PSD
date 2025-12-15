
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import folium
from streamlit_folium import st_folium

st.title("🌫️ Prediksi Konsentrasi NO₂ Surabaya (7 Hari ke Depan)")

uploaded_file = st.file_uploader("Upload file CSV NO₂", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Perbaikan bug jika file CSV memiliki header aneh
    if df.columns[0] == "Tanggal,NO2":
        # Reset file pointer dan baca ulang
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, names=["Tanggal", "NO2"], skiprows=1)

    # Perbaikan dari error Anda sebelumnya (time vs Tanggal)
    if 'time' in df.columns:
        df.rename(columns={'time': 'Tanggal'}, inplace=True)

    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['NO2'] = pd.to_numeric(df['NO2'], errors='coerce')

    # Mengisi nilai hilang (penting untuk model)
    df['NO2'] = df['NO2'].interpolate(method='linear')
    df = df.dropna(subset=['Tanggal','NO2']).sort_values('Tanggal')

    st.line_chart(df.set_index('Tanggal')['NO2'])

    model = SARIMAX(df['NO2'], order=(1,1,1), seasonal_order=(1,1,1,7))
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=7)
    pred_mean = forecast.predicted_mean
    future_dates = [df['Tanggal'].iloc[-1] + timedelta(days=i) for i in range(1,8)]
    pred_df = pd.DataFrame({'Tanggal': future_dates, 'Prediksi_NO2': pred_mean.values})

    st.subheader("📅 Prediksi 7 Hari ke Depan")
    st.dataframe(pred_df)

    plt.figure(figsize=(10,5))
    plt.plot(df.set_index('Tanggal')['NO2'], label='Aktual', color='blue')
    plt.plot(pred_df.set_index('Tanggal')['Prediksi_NO2'], color='red', marker='o', label='Prediksi')
    plt.legend()
    st.pyplot(plt)

    lat, lon = -7.2575, 112.7521
    m = folium.Map(location=[lat, lon], zoom_start=11)
    mean_pred = pred_df['Prediksi_NO2'].mean()
    folium.CircleMarker([lat, lon], radius=15, color='red', fill=True,
                        popup=f"Prediksi rata-rata NO₂: {mean_pred:.2e}").add_to(m)
    st_folium(m, width=700)
