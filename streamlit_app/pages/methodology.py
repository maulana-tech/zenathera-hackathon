import streamlit as st
from utils import setup_page

def show():
    st.title("Metodologi & Pendekatan Teknis")
    
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, fillcolor="#e0e7ff", fontname="Inter"];
        
        Scraping [label="Data Scraping\n(Twitter, TikTok,\nYouTube, Play Store)"];
        Preprocessing [label="Preprocessing\n(Cleaning, Stemming)"];
        Sentiment [label="Sentiment Analysis\n(IndoBERT)"];
        Topic [label="Topic Modeling\n(BERTopic)"];
        Dashboard [label="Streamlit\nDashboard", fillcolor="#4f46e5", fontcolor="white"];
        
        Scraping -> Preprocessing;
        Preprocessing -> Sentiment;
        Preprocessing -> Topic;
        Sentiment -> Dashboard;
        Topic -> Dashboard;
    }
    """)
    
    st.markdown("""
    ### 1. Pengumpulan Data (Data Scraping)
    Kami mengumpulkan **9.000+ data** ulasan dan komentar dari 4 platform digital utama untuk mendapatkan perspektif yang komprehensif:
    
    - **Twitter (X)**: Menangkap percakapan real-time dan keluhan langsung ke akun resmi.
    - **TikTok**: Mengambil komentar dari video-video viral terkait CoreTax.
    - **YouTube**: Menganalisis diskusi mendalam di kolom komentar video edukasi/berita pajak.
    - **Google Play Store**: Mengumpulkan ulasan pengguna aplikasi mobile (M-Pajak/CoreTax).
    
    ### 2. Preprocessing Data
    Tahapan pembersihan data meliputi:
    - **Cleaning**: Menghapus URL, mention, hashtag, dan karakter spesial.
    - **Case Folding**: Mengubah teks menjadi huruf kecil.
    - **Normalisasi Slang**: Mengubah kata tidak baku (gaul) menjadi baku menggunakan kamus khusus.
    - **Stemming**: Mengubah kata berimbuhan menjadi kata dasar (menggunakan Sastrawi/mpstemmer).
    
    ### 3. Analisis Sentimen (RoBERTa)
    Menggunakan model **IndoBERT (RoBERTa)** yang telah dilatih (fine-tuned) untuk analisis sentimen bahasa Indonesia (`w11wo/indonesian-roberta-base-sentiment-classifier`).
    - **InSet Lexicon**: Kamus sentimen bahasa Indonesia.
    - **IndoBERT**: Model Transformer untuk klasifikasi sentimen yang lebih akurat.
    - **Akurasi Tinggi**: Model transformer mampu memahami konteks kalimat lebih baik daripada metode tradisional.
    - **Output**: Label (Positif, Negatif, Netral) dan Skor Keyakinan.
    
    ### 4. Pemodelan Topik (BERTopic)
    Untuk memahami "apa yang dibicarakan", kami menggunakan **BERTopic**:
    - **Embedding**: `distiluse-base-multilingual-cased-v2` untuk mengubah teks menjadi vektor.
    - **Clustering**: Mengelompokkan vektor yang mirip untuk menemukan topik tersembunyi.
    - **Representation**: Mengekstrak kata kunci (c-TF-IDF) untuk setiap topik.
    """)
    
    st.info("Pendekatan ini memungkinkan kita tidak hanya tahu 'apakah user marah', tapi juga 'kenapa mereka marah'.")

if __name__ == "__main__":
    setup_page("Methodology")
    show()
