# CoreTax Sentiment Analysis

Project ini bertujuan untuk menganalisis sentimen publik terhadap sistem CoreTax DJP menggunakan data dari YouTube, Play Store, dan Media Sosial.

## Struktur Folder

```
project/
├── data/           -> Dataset mentah (raw) dan hasil preprocessing (processed)
│   ├── processed/  -> Data yang sudah dibersihkan dan digabungkan
│   └── ...         -> Data mentah (CSV scraping)
├── notebooks/      -> Jupyter Notebooks untuk analisis dan eksperimen
│   └── Hackathon Sentiment Analysis Improved.ipynb
├── src/            -> Source code utama dan script utilitas
│   ├── crawl_twitter.py    -> Script untuk crawling data Twitter
│   └── verify_integration.py
├── models/         -> Tempat menyimpan model IndoBERT yang sudah dilatih
├── outputs/        -> Hasil analisis (plot, visualisasi) dan laporan
│   └── presentation_outline.md
├── requirements.txt -> Daftar library Python yang dibutuhkan
└── README.md       -> Dokumentasi project ini
```

## Cara Menjalankan

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Catatan: Untuk crawling Twitter, pastikan Node.js sudah terinstall karena menggunakan `tweet-harvest`.*

2.  **Crawling Data Twitter (Opsional):**
    Jika ingin mengambil data terbaru dari Twitter:
    *   Buka `src/crawl_twitter.py` dan update `TWITTER_AUTH_TOKEN` dengan token Anda.
    *   Jalankan script:
        ```bash
        python src/crawl_twitter.py
        ```
    *   Data akan otomatis tersimpan di folder `data/`.

3.  **Jalankan Notebook Analisis:**
    *   Buka `notebooks/Hackathon Sentiment Analysis Improved.ipynb`.
    *   Jalankan semua cell. Notebook akan:
        *   Memuat data dari folder `data/`.
        *   Melakukan preprocessing dan labeling sentimen.
        *   Melatih model IndoBERT (disimpan ke `models/`).
        *   Menghasilkan visualisasi dan rekomendasi (disimpan ke `outputs/`).

4.  **Verifikasi Data:**
    Anda dapat menjalankan script verifikasi untuk mengecek integritas data:
    ```bash
    cd src
    python verify_integration.py
    ```

## Fitur Utama
- **Data Integration:** Menggabungkan data dari berbagai sumber.
- **Sentiment Analysis:** Menggunakan RoBERTa (`w11wo/indonesian-roberta-base-sentiment-classifier`).
- **Advanced Visualization:** WordCloud, N-grams, Co-occurrence Network.
- **Automated Insights:** Rekomendasi perbaikan berdasarkan analisis sentimen negatif.
