# CoreTax Sentiment Analysis - Presentation Outline

## Topic 1: Title Topic
- **Title:** Analisis Sentimen Publik terhadap CoreTax System
- **Subtitle:** Mengungkap Persepsi Pengguna melalui Data Multi-Platform (YouTube, Play Store, Twitter/TikTok)
- **Presenter:** [Nama Tim/Anda]
- **Date:** [Tanggal]

## Topic 2: Executive Summary
- **Goal:** Memahami sentimen masyarakat terhadap sistem CoreTax DJP yang baru.
- **Key Findings:**
    - Dominasi sentimen [Positif/Negatif/Netral] (berdasarkan hasil notebook).
    - Isu utama: Login gagal, Kode verifikasi, Koneksi lambat.
    - Platform dengan keluhan terbanyak: Play Store, YouTube, Twitter/TikTok.
- **Outcome:** Rekomendasi perbaikan teknis dan strategi komunikasi.

## Topic 3: Latar Belakang & Tujuan
- **Context:** Peluncuran CoreTax sebagai sistem inti administrasi perpajakan baru.
- **Problem:** Perlunya monitoring feedback publik secara real-time untuk identifikasi bug dan kepuasan pengguna.
- **Objectives:**
    1. Mengintegrasikan data ulasan dari berbagai sumber.
    2. Melakukan klasifikasi sentimen otomatis (RoBERTa).
    3. Memberikan rekomendasi berbasis data (Data-Driven Insights).

## Topic 4: Data Sources & Integration
- **Total Data:** ~7,000+ ulasan unik.
- **Sources:**
    - **Play Store:** Ulasan aplikasi mobile (Rating & Komentar).
    - **YouTube:** Komentar pada video sosialisasi/berita CoreTax.
    - **Social Media (Twitter/TikTok):** Diskusi publik dan keluhan langsung.
- **Process:** Scraping -> Cleaning -> Deduplication -> Integration.

## Topic 5: Methodology (AI Pipeline)
- **Preprocessing:** Cleaning (Regex), Lowercasing, Stemming (Sastrawi).
- **Sentiment Labeling:**
    - Model: `w11wo/indonesian-roberta-base-sentiment-classifier`.
    - Classes: Positive, Negative, Neutral.
- **Advanced Modeling:**
    - **IndoBERT Fine-tuning:** Melatih model khusus untuk domain pajak.
    - **TF-IDF:** Ekstraksi kata kunci penting per kategori sentimen.

## Topic 6: Analisis Sentimen (Overview)
- **Visual:** Pie Chart (Proporsi Sentimen).
- **Insight:**
    - Berapa persen pengguna yang puas vs tidak puas?
    - Apakah sentimen netral (pertanyaan/diskusi) mendominasi?
- **Visual:** Box Plot (Confidence Score).
    - Seberapa yakin model terhadap prediksinya?

## Topic 7: Analisis Temporal & Sumber
- **Visual:** Line Chart (Tren Sentimen Bulanan).
    - Kapan lonjakan keluhan terjadi? (Misal: Saat peluncuran fitur baru).
- **Visual:** Bar Chart (Sentimen per Sumber).
    - Apakah pengguna YouTube lebih positif dibanding pengguna Play Store?
    - **Insight:** Karakteristik audiens tiap platform berbeda.

## Topic 8: Deep Dive - Apa Kata Mereka?
- **Visual:** Word Clouds (Positif vs Negatif).
    - **Positif:** "Membantu", "Mudah", "Cepat".
    - **Negatif:** "Gagal", "Error", "Login", "OTP".
- **Visual:** N-grams (Bigram/Trigram).
    - Frasa umum: "Kode verifikasi", "Gagal login", "Tidak bisa".

## Topic 9: Advanced Insights
- **Visual:** Co-occurrence Network.
    - Hubungan antar kata (misal: "Login" terhubung kuat dengan "Gagal" dan "Jaringan").
- **Visual:** TF-IDF Top Keywords.
    - Kata-kata unik yang menjadi ciri khas tiap sentimen.

## Topic 10: Model Performance (IndoBERT)
- **Visual:** Confusion Matrix.
    - Evaluasi akurasi model fine-tuning.
    - Dimana model sering salah prediksi?
- **Visual:** Training Loss Curve.
    - Bukti proses pembelajaran model yang stabil.

## Topic 11: Rekomendasi (Actionable Insights)
- **Tabel Rekomendasi Otomatis:**
    | Isu (Bigram) | Frekuensi | Rekomendasi Teknis |
    | :--- | :--- | :--- |
    | "Gagal Login" | Tinggi | Audit stabilitas server auth & error messaging. |
    | "Kode Verifikasi" | Sedang | Cek latency gateway SMS/Email OTP. |
    | "Aplikasi Berat" | Sedang | Optimasi performa aplikasi mobile. |

## Topic 12: Kesimpulan & Future Work
- **Kesimpulan:** Sentimen publik memberikan sinyal kuat area perbaikan teknis.
- **Future Work:**
    - Integrasi Dashboard Real-time.
    - Aspect-Based Sentiment Analysis (ABSA) untuk deteksi fitur spesifik.
    - Automasi pelaporan tiket support dari sentimen negatif.

## Topic 13: Q&A
- Terima Kasih.
