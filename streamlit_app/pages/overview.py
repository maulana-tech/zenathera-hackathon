import streamlit as st
from utils import setup_page, load_data
import plotly.express as px
import os

def show():
    setup_page("Overview")

    st.title("CoreTax Sentiment Overview")
    st.markdown("### Public Perception Dashboard")
    
    df = load_data()
    if df.empty:
        st.warning("No data available.")
        return

    # Metrics
    total_comments = len(df)
    sentiment_counts = df['sentiment'].value_counts()
    neg_pct = (sentiment_counts.get('negative', 0) / total_comments) * 100
    pos_pct = (sentiment_counts.get('positive', 0) / total_comments) * 100
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card metric-card">
            <div class="metric-label">Total Komentar</div>
            <div class="metric-value primary">{total_comments:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card metric-card">
            <div class="metric-label">Sentimen Negatif</div>
            <div class="metric-value negative">{neg_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card metric-card">
            <div class="metric-label">Sentimen Positif</div>
            <div class="metric-value positive">{pos_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Charts
    st.markdown("---")
    c_chart1, c_chart2 = st.columns(2)
    
    with c_chart1:
        st.subheader("Distribusi Sentimen (Interaktif)")
        fig_pie = px.pie(
            names=sentiment_counts.index, 
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map={'positive': '#10b981', 'negative': '#f43f5e', 'neutral': '#94a3b8'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with c_chart2:
        st.subheader("Distribusi per Platform (Interaktif)")
        source_counts = df.groupby(['source', 'sentiment']).size().reset_index(name='count')
        fig_bar = px.bar(
            source_counts, 
            x='source', 
            y='count', 
            color='sentiment',
            color_discrete_map={'positive': '#10b981', 'negative': '#f43f5e', 'neutral': '#94a3b8'},
            barmode='group'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Static Visualizations from Outputs
    st.markdown("---")
    st.header("Visualisasi Statis & Analisis Lanjutan")
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        st.markdown("**Distribusi Sentimen per Sumber**")
        if os.path.exists("outputs/distribusi-sentiment-setiap-sumber.png"):
            st.image("outputs/distribusi-sentiment-setiap-sumber.png", use_column_width=True)
        else:
            st.info("Gambar distribusi sentimen tidak tersedia.")
            
    with col_img2:
        st.markdown("**Top 20 Kata Kunci (TF-IDF)**")
        if os.path.exists("outputs/top-20-TF-IDF-Score.png"):
            st.image("outputs/top-20-TF-IDF-Score.png", use_column_width=True)
        else:
            st.info("Gambar TF-IDF tidak tersedia.")

    # Technical Implementation Details
    st.markdown("---")
    st.header("Teknis Implementasi (AI Pipeline)")
    
    with st.expander("Lihat Detail Teknis", expanded=True):
        st.markdown("""
        **1. Preprocessing**
        - **Cleaning**: Regex untuk menghapus noise.
        - **Lowercasing**: Normalisasi huruf kecil.
        - **Stemming**: Menggunakan library **Sastrawi**.
        
        **2. Sentiment Labeling**
        - **Model**: `w11wo/indonesian-roberta-base-sentiment-classifier`.
        - **Classes**: Positive, Negative, Neutral.
        - **Fine-tuning**: Dilakukan untuk domain spesifik pajak.
        
        **3. Advanced Modeling**
        - **TF-IDF**: Ekstraksi kata kunci penting per kategori sentimen.
        - **BERTopic**: Topic modeling untuk menemukan cluster permasalahan.
        
        **4. Data Sources**
        - **Total Data**: ~9,000+ ulasan unik.
        - **Sources**: Play Store, YouTube, Twitter/TikTok.
        """)
        
    # Recent Comments
    st.markdown("---")
    st.subheader("Sampel Komentar Terbaru")
    st.dataframe(df[['source', 'sentiment', 'text']].head(10), use_container_width=True)

if __name__ == "__main__":
    show()
