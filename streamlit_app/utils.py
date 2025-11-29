import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_data():
    """Loads the CoreTax sentiment data from CSV."""
    # Adjust path relative to where streamlit is run (root)
    file_path = "models/BERTopic-CoreTax-data.csv"
    if not os.path.exists(file_path):
        st.error(f"Data file not found at {file_path}")
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    # Ensure text column exists (renaming if needed based on previous steps)
    if 'text' not in df.columns and 'hasil normalisasi' in df.columns:
        df['text'] = df['hasil normalisasi']
        
    return df

def local_css(file_name):
    """Injects custom CSS from a file."""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def setup_page(title="CoreTax Sentiment"):
    """Common setup for all pages."""
    st.set_page_config(
        page_title=title,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    local_css("streamlit_app/style.css")
    
    # Sidebar UX Improvements
    with st.sidebar:
        st.markdown("### 📊 CoreTax Dashboard")
        
        # Navigation is handled automatically by Streamlit pages
        
        st.markdown("---")
        
        with st.expander("ℹ️ Tentang Aplikasi", expanded=False):
            st.markdown("""
            Dashboard ini menyajikan analisis sentimen publik terhadap sistem **CoreTax** berdasarkan data media sosial.
            
            **Fitur:**
            - Analisis Sentimen (RoBERTa)
            - Pemodelan Topik (BERTopic)
            - Rekomendasi Strategis
            """)
            
        with st.expander("⚙️ Pengaturan Tampilan", expanded=False):
            st.caption("Sesuaikan preferensi tampilan Anda.")
            # Note: Real theme toggling requires more complex state or custom component, 
            # but we can put placeholders or simple session state toggles here.
            st.checkbox("Tampilkan Detail Metrik", value=True)
        
        st.markdown("---")
        st.caption("© 2025 Zenithera Hackathon")

def load_html_asset(filename):
    """Loads an HTML asset content."""
    path = os.path.join("streamlit_app/assets", filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return None
