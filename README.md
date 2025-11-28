# HACKATHON ZENITHERA - KELOMPOK 5

## Judul : Sentiment Analysis sebagai Dasar Pengambilan Keputusan Peningkatan Layanan CoreTax pada Sistem Perpajakan Digital.

### Anggota Kelompok :

1. Muhammad Iqbal
2. Muhammad Maulana Firdaussyah

![Banner](outputs/wordcloud-sentiment.png)

### Deskripsi Project

Project ini dikembangkan untuk **Hackathon Zenithera** oleh **Kelompok 5**. Tujuan utamanya adalah melakukan analisis mendalam terhadap sentimen publik mengenai implementasi sistem **CoreTax Administration System (CTAS)** oleh Direktorat Jenderal Pajak (DJP).

Kami mengumpulkan data dari berbagai platform digital (**YouTube, Play Store, Twitter/X, dan TikTok**) untuk menangkap persepsi wajib pajak secara komprehensif. Analisis ini tidak hanya mengklasifikasikan sentimen (Positif, Negatif, Netral) menggunakan model Deep Learning (**RoBERTa**), tetapi juga menggali topik permasalahan utama di balik sentimen negatif menggunakan **BERTopic**, serta memberikan rekomendasi berbasis data untuk peningkatan layanan.

### Tech Stack & Tools

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-orange?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![NodeJS](https://img.shields.io/badge/node.js-6DA55F?style=for-the-badge&logo=node.js&logoColor=white)

Project ini dibangun menggunakan teknologi dan library berikut:

*   **Language:** Python 3.8+
*   **Data Processing:** Pandas, NumPy
*   **NLP & Machine Learning:**
    *   **Transformers (Hugging Face):** Menggunakan model `w11wo/indonesian-roberta-base-sentiment-classifier` untuk akurasi tinggi dalam analisis sentimen Bahasa Indonesia.
    *   **BERTopic:** Untuk topic modeling tingkat lanjut guna menemukan cluster permasalahan.
    *   **Scikit-learn:** Untuk TF-IDF dan preprocessing tambahan.
    *   **Sastrawi:** Library khusus untuk stemming Bahasa Indonesia.
*   **Visualization:** Matplotlib, Seaborn, WordCloud, NetworkX.
*   **Scraping:** Tweet-Harvest (Node.js/Playwright) untuk pengambilan data Twitter.

### Struktur Folder

```
project/
├── data/                      # Dataset mentah dan hasil proses
│   ├── processed/             # Data hasil cleaning & preprocessing
│   │   ├── CoreTax Preprocessing Results.csv     # Output preprocessing
│   ├── Data-Scrape-PlayStore.csv                 # Raw data Play Store
│   ├── Data-Scrape-YouTube.csv                   # Raw data YouTube
│   ├── Data-Combined-Twitter-Tiktok.csv          # Raw data Twitter & TikTok
│   └── kamuskatabaku.xlsx                        # Kamus normalisasi teks
│
├── notebooks/                 # Jupyter Notebooks untuk eksperimen
│   ├── Final_Zenithera_Hackatoon.ipynb           # Notebook utama (Final)
│   └── Zenithera_Analysis-v1.ipynb               # Notebook versi awal
│
├── src/                       # Source code modular
│   ├── scraping/              # Script crawling data
│   │   └── crawl_twitter.py                      # Script crawl Twitter
│   ├── main.py                # Entry point pipeline
│   ├── config.py              # Konfigurasi path & konstanta
│   ├── data_loader.py         # Modul load & merge data
│   ├── preprocessing.py       # Modul cleaning & stemming
│   ├── sentiment_analysis.py  # Modul RoBERTa labeling
│   ├── visualization.py       # Modul plotting & WordCloud
│   └── topic_modeling.py      # Modul BERTopic analysis
│
├── models/                    # Model yang telah dilatih
│   └── BERTopic-CoreTax-data.csv             # Folder model BERTopic
│
├── outputs/                   # Hasil visualisasi & laporan
│   ├── distribusi-sentiment-setiap-sumber.png    # Plot distribusi sentimen
│   ├── sentiment-coretax.png                     # Plot total sentimen
│   ├── wordcloud-sentiment.png                   # Visualisasi WordCloud
│   ├── intertopic-distance-map.png               # Visualisasi topik
│   └── ...                                       # File output lainnya
│
├── requirements.txt           # Daftar library Python
└── README.md                  # Dokumentasi project
```

### Cara Menjalankan

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Untuk crawling Twitter, pastikan Node.js sudah terinstall.*

2.  **Jalankan Pipeline Analisis (Python Script):**
    Untuk menjalankan seluruh proses (Load -> Preprocess -> Sentiment -> Viz -> Topic Modeling):
    ```bash
    python -m src.main
    ```
    *Pastikan dijalankan dari root folder project.*

3.  **Crawling Data Twitter (Opsional):**
    ```bash
    python src/scraping/crawl_twitter.py
    ```

4.  **Jalankan Notebook (Eksperimen):**
    Buka `notebooks/Final_Zenithera_Hackatoon.ipynb` untuk analisis interaktif.

### Fitur Utama
- **Modular Codebase:** Kode dipecah menjadi modul-modul terpisah di `src/` agar lebih rapi dan mudah di-maintain.
- **Data Integration:** Menggabungkan data dari berbagai sumber.
- **Sentiment Analysis:** Menggunakan RoBERTa (`w11wo/indonesian-roberta-base-sentiment-classifier`).
- **Advanced Visualization:** WordCloud, N-grams, Co-occurrence Network.
- **Topic Modeling:** Menggunakan BERTopic untuk menemukan topik utama dalam sentimen negatif.

### Hasil Analisis (Preview)

Berikut adalah beberapa contoh visualisasi yang dihasilkan oleh project ini:

#### 1. Distribusi Sentimen
![Sentiment Distribution](outputs/sentiment-coretax.png)

#### 2. Hierarchical Clustering
![Hierarchical Clustering](outputs/hierarchical-clustering.png)

#### 3. Topik Word Scores (BERTopic)
![Topic Word Scores](outputs/topic-word-scores.png)

*Untuk hasil lengkap, silakan cek folder `outputs/`.*
