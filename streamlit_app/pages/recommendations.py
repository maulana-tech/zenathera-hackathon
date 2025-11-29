import streamlit as st
from utils import setup_page, load_data

def show():
    setup_page("Recommendations")

    st.title("Rekomendasi & Tindak Lanjut")
    
    df = load_data()
    
    # Data Driven Validation
    st.header("Validasi Berbasis Data")
    if not df.empty:
        login_issues = df[df['text'].str.contains('login|masuk|gagal', case=False, na=False)].shape[0]
        error_issues = df[df['text'].str.contains('error|bug|rusak', case=False, na=False)].shape[0]
        slow_issues = df[df['text'].str.contains('lemot|lambat|berat', case=False, na=False)].shape[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Keluhan Login", f"{login_issues} user", delta="High Priority", delta_color="inverse")
        c2.metric("Laporan Error", f"{error_issues} user", delta="Medium Priority", delta_color="inverse")
        c3.metric("Isu Performa", f"{slow_issues} user", delta="Medium Priority", delta_color="inverse")
    
    st.markdown("---")
    st.header("Rekomendasi Strategis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.subheader("🔴 Prioritas Tinggi (Teknis)")
            st.info("""
            **1. Stabilisasi Sistem Login**
            - **Masalah**: Kegagalan autentikasi berulang.
            - **Solusi**: Audit session management, perbaiki pesan error yang tidak jelas, dan cek gateway OTP.
            
            **2. Optimasi Performa Mobile**
            - **Masalah**: Aplikasi dianggap berat dan lambat.
            - **Solusi**: Lakukan *profiling* aplikasi, kurangi *load time* aset, dan optimalkan *API response time*.
            """)
            
    with col2:
        with st.container():
            st.subheader("🟡 Prioritas Menengah (UX & Edukasi)")
            st.warning("""
            **3. Simplifikasi Navigasi**
            - **Masalah**: Menu terlalu kompleks bagi wajib pajak awam.
            - **Solusi**: Tambahkan fitur *Guided Tour* saat pertama kali login dan sederhanakan hierarki menu.
            
            **4. Strategi Komunikasi Proaktif**
            - **Masalah**: Kebingungan pengguna saat error terjadi.
            - **Solusi**: Gunakan bahasa manusiawi pada pesan error (bukan kode teknis) dan buat konten edukasi video pendek (TikTok/Reels).
            """)

    st.markdown("---")
    st.subheader("📈 Rencana Monitoring")
    st.markdown("""
    - **Mingguan**: Tracking jumlah keluhan "Login" dan "Error" setelah patch perbaikan.
    - **Bulanan**: Analisis ulang sentimen untuk melihat tren kepuasan jangka panjang.
    - **Kuartalan**: Survei kepuasan pengguna terintegrasi di dalam aplikasi.
    """)
    
    st.success("💡 **Goal Akhir**: Mengubah sentimen negatif menjadi kepercayaan publik melalui perbaikan yang responsif.")

if __name__ == "__main__":
    show()
