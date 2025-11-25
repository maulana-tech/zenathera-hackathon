# ============================================================================
# 1. SETUP DAN INSTALASI
# ============================================================================
print("=" * 80)
print("STEP 1: SETUP DAN INSTALASI")
print("=" * 80)

# Install required packages
!pip install -q transformers torch pandas numpy scikit-learn matplotlib seaborn wordcloud

!pip install -q indobenchmark-toolkit
!pip install -q Sastrawi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import warnings
import requests
from io import StringIO
warnings.filterwarnings('ignore')

# Import untuk preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Import untuk IndoBERT
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Import Sastrawi untuk stemming
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("✓ Instalasi selesai!")

# ============================================================================
# 2. DOWNLOAD INSET LEXICON DARI GITHUB
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DOWNLOAD INSET LEXICON DARI GITHUB")
print("=" * 80)

# Download InSet Lexicon files
inset_base_url = "https://raw.githubusercontent.com/fajri91/InSet/master/"

print("Downloading InSet Lexicon...")

# Download positive words
positive_url = inset_base_url + "positive.tsv"
negative_url = inset_base_url + "negative.tsv"

try:
    # Download positive lexicon
    response_pos = requests.get(positive_url)
    positive_lexicon = {}
    for line in response_pos.text.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            word = parts[0].strip()
            try:
                weight = float(parts[1].strip())
                positive_lexicon[word] = weight
            except:
                pass
    
    # Download negative lexicon
    response_neg = requests.get(negative_url)
    negative_lexicon = {}
    for line in response_neg.text.strip().split('\n'):
        parts = line.split('\t')
        if len(parts) >= 2:
            word = parts[0].strip()
            try:
                weight = float(parts[1].strip())
                negative_lexicon[word] = weight
            except:
                pass
    
    print(f"✓ InSet Lexicon berhasil didownload!")
    print(f"  - Positive words: {len(positive_lexicon)} kata")
    print(f"  - Negative words: {len(negative_lexicon)} kata")
    print(f"  - Total: {len(positive_lexicon) + len(negative_lexicon)} kata")
    
except Exception as e:
    print(f"⚠️ Error downloading InSet Lexicon: {e}")
    print("Menggunakan lexicon backup...")
    
    # Backup lexicon (subset dari InSet)
    positive_lexicon = {
        'bagus': 3, 'baik': 3, 'senang': 4, 'suka': 3, 'hebat': 4,
        'mantap': 4, 'keren': 4, 'indah': 3, 'cantik': 3, 'pintar': 3,
        'cerdas': 3, 'sukses': 4, 'berhasil': 4, 'menang': 4, 'juara': 5,
        'terbaik': 5, 'sempurna': 5, 'luar biasa': 5, 'fantastis': 5,
        'mengagumkan': 5, 'membantu': 3, 'berguna': 3, 'bermanfaat': 3,
        'positif': 3, 'optimis': 3, 'semangat': 3, 'gembira': 4,
        'bahagia': 4, 'puas': 3, 'setuju': 2, 'mendukung': 3, 'oke': 2,
        'benar': 2, 'tepat': 2, 'cocok': 2, 'recommended': 3, 'top': 4
    }
    
    negative_lexicon = {
        'buruk': -3, 'jelek': -3, 'tidak': -1, 'bukan': -1, 'sedih': -3,
        'kecewa': -4, 'marah': -4, 'benci': -5, 'kesal': -3, 'gagal': -4,
        'kalah': -3, 'rugi': -3, 'salah': -2, 'error': -2, 'rusak': -3,
        'hancur': -5, 'parah': -4, 'payah': -3, 'lemah': -2, 'bodoh': -4,
        'korup': -5, 'korupsi': -5, 'curang': -4, 'bohong': -4, 'tipu': -4,
        'penipuan': -5, 'scam': -5, 'fake': -3, 'palsu': -3, 'sampah': -4,
        'busuk': -4, 'menyebalkan': -4, 'mengganggu': -3, 'mengecewakan': -4,
        'zonk': -3, 'ngaco': -3, 'tolol': -4, 'goblok': -5, 'kampret': -5
    }

# ============================================================================
# 3. FUNGSI PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: SETUP PREPROCESSING FUNCTIONS")
print("=" * 80)

# Initialize Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()

# Tambahan stopwords khusus
custom_stopwords = set([
    'yg', 'dgn', 'nya', 'nih', 'sih', 'dong', 'deh', 'aja', 'loh', 'kalo',
    'gak', 'gk', 'ga', 'nggak', 'ngga', 'enggak', 'tdk', 'tdk', 'gue', 'gw',
    'ane', 'ente', 'lu', 'loe', 'kamu', 'aku', 'saya', 'kita', 'kami',
    'rt', 'via', 'cc', 'dm', 'pm', 'follow', 'followers', 'following'
])
stopwords = set(stopwords).union(custom_stopwords)

# Slang words dictionary untuk normalisasi
slang_dict = {
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'nggak': 'tidak',
    'ngga': 'tidak', 'enggak': 'tidak', 'gue': 'saya', 'gw': 'saya',
    'ane': 'saya', 'lu': 'kamu', 'loe': 'kamu', 'ente': 'kamu',
    'bgt': 'banget', 'bgt': 'banget', 'bgt': 'banget', 'bgt': 'banget',
    'yg': 'yang', 'dgn': 'dengan', 'utk': 'untuk', 'krn': 'karena',
    'tp': 'tetapi', 'tapi': 'tetapi', 'klo': 'kalau', 'kalo': 'kalau',
    'gmn': 'bagaimana', 'gimana': 'bagaimana', 'knp': 'kenapa',
    'kenapa': 'mengapa', 'emg': 'memang', 'emang': 'memang',
    'udh': 'sudah', 'udah': 'sudah', 'dah': 'sudah', 'blm': 'belum',
    'blom': 'belum', 'belom': 'belum', 'jg': 'juga', 'jgn': 'jangan',
    'jgn': 'jangan', 'hrs': 'harus', 'msh': 'masih', 'lg': 'lagi',
    'bkn': 'bukan', 'trs': 'terus', 'trus': 'terus', 'bs': 'bisa',
    'bisa': 'dapat', 'org': 'orang', 'orng': 'orang', 'org': 'orang',
    'byk': 'banyak', 'bnyk': 'banyak', 'bgt': 'banget', 'bgt': 'banget',
    'sy': 'saya', 'km': 'kamu', 'kl': 'kalau', 'dl': 'dahulu',
    'skrg': 'sekarang', 'skrng': 'sekarang', 'skg': 'sekarang',
    'thn': 'tahun', 'bln': 'bulan', 'mggu': 'minggu', 'hr': 'hari',
    'mnt': 'menit', 'dtk': 'detik', 'jm': 'jam', 'wkt': 'waktu',
    'spt': 'seperti', 'sprt': 'seperti', 'krg': 'kurang', 'lbh': 'lebih',
    'sm': 'sama', 'dr': 'dari', 'pd': 'pada', 'dl': 'dalam',
    'dlm': 'dalam', 'stlh': 'setelah', 'sblm': 'sebelum', 'sdh': 'sudah',
    'tdk': 'tidak', 'tdk': 'tidak', 'blh': 'boleh', 'hrs': 'harus',
    'pny': 'punya', 'pnya': 'punya', 'pngen': 'ingin', 'pgn': 'ingin',
    'mksh': 'terima kasih', 'thx': 'terima kasih', 'thanks': 'terima kasih',
    'thankyou': 'terima kasih', 'ty': 'terima kasih'
}

def normalize_slang(text):
    """Normalisasi kata-kata slang/gaul"""
    words = text.split()
    normalized = []
    for word in words:
        normalized.append(slang_dict.get(word.lower(), word))
    return ' '.join(normalized)

def clean_text_twitter(text):
    """Preprocessing khusus untuk data Twitter"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove RT (retweet)
    text = re.sub(r'\brt\b', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username) - simpan untuk analisis tapi hapus dari text
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags tapi simpan kata-katanya
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove email
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove angka
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters dan punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize slang words
    text = normalize_slang(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    text = ' '.join(words)
    
    # Stemming
    text = stemmer.stem(text)
    
    return text

def clean_text_tiktok(text):
    """Preprocessing khusus untuk data TikTok"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags tapi simpan kata-katanya
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove emoji patterns (TikTok banyak emoji)
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    
    # Remove angka
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize slang words
    text = normalize_slang(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stopwords and len(w) > 2]
    text = ' '.join(words)
    
    # Stemming
    text = stemmer.stem(text)
    
    return text

print("✓ Preprocessing functions ready!")

# ============================================================================
# 4. FUNGSI LABELING DENGAN INSET LEXICON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: SETUP LABELING FUNCTION")
print("=" * 80)

def inset_sentiment_labeling(text, positive_lex, negative_lex, threshold=0.5):
    """
    Labeling sentimen menggunakan InSet Lexicon dengan weighted scoring
    
    Parameters:
    - text: teks yang sudah di-preprocessing
    - positive_lex: dictionary kata positif dengan bobot
    - negative_lex: dictionary kata negatif dengan bobot
    - threshold: threshold untuk menentukan neutral
    
    Returns:
    - sentiment: 'positive', 'negative', atau 'neutral'
    - score: skor sentimen
    """
    if not text or len(text.strip()) == 0:
        return 'neutral', 0.0
    
    words = text.lower().split()
    
    # Hitung weighted score
    pos_score = sum(positive_lex.get(word, 0) for word in words)
    neg_score = sum(abs(negative_lex.get(word, 0)) for word in words)
    
    # Total score
    total_score = pos_score - neg_score
    
    # Normalisasi berdasarkan panjang text
    if len(words) > 0:
        normalized_score = total_score / len(words)
    else:
        normalized_score = 0
    
    # Tentukan sentiment
    if normalized_score > threshold:
        return 'positive', normalized_score
    elif normalized_score < -threshold:
        return 'negative', normalized_score
    else:
        return 'neutral', normalized_score

print("✓ Labeling function ready!")

# ============================================================================
# 5. MOUNT GOOGLE DRIVE DAN LOAD DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: MOUNT GOOGLE DRIVE DAN LOAD DATA")
print("=" * 80)

# Mount Google Drive
from google.colab import drive
import os

print("Mounting Google Drive...")
drive.mount('/content/drive/')
print("✓ Google Drive mounted!")

# Set data path
DATA_PATH = '/content/drive/MyDrive/Hackathon/data/'
print(f"Data path: {DATA_PATH}")

# ============================================================================
# 6. LOAD DAN PREPROCESSING DATA TWITTER
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: LOAD DAN PREPROCESSING DATA TWITTER")
print("=" * 80)

# Load Twitter data
try:
    print("Loading Twitter data...")
    df_twitter = pd.read_csv(
        DATA_PATH + 'CoreTax-Twitter-01.csv',
        header=None,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    print(f"✓ Twitter data loaded: {len(df_twitter)} baris")
    print(f"Kolom yang tersedia: {df_twitter.columns.tolist()}")
    
    # Kolom ke-4 (indeks 3) diasumsikan sebagai teks tweet
    df_twitter = df_twitter[[3]].rename(columns={3: 'text'})
    df_twitter['source'] = 'Twitter'
    
    print(f"✓ Menggunakan kolom indeks 3 sebagai 'text'")
    print(f"Preview data Twitter:")
    print(df_twitter.head())
    
except Exception as e:
    print(f"⚠️ Error loading Twitter data: {e}")
    print("Creating empty Twitter dataframe...")
    df_twitter = pd.DataFrame({'text': [], 'source': []})

# Preprocessing Twitter
if len(df_twitter) > 0:
    print("\nMemproses data Twitter...")
    df_twitter['cleaned_text'] = df_twitter['text'].apply(clean_text_twitter)
    df_twitter = df_twitter[df_twitter['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Labeling dengan InSet
    print("Melakukan labeling sentimen...")
    sentiments = []
    scores = []
    for text in df_twitter['cleaned_text']:
        sentiment, score = inset_sentiment_labeling(text, positive_lexicon, negative_lexicon)
        sentiments.append(sentiment)
        scores.append(score)
    
    df_twitter['sentiment'] = sentiments
    df_twitter['sentiment_score'] = scores
    df_twitter['source'] = 'twitter'
    
    print(f"✓ Twitter preprocessing selesai!")
    print(f"Data setelah cleaning: {len(df_twitter)} baris")
    print(f"\nDistribusi sentimen Twitter:")
    print(df_twitter['sentiment'].value_counts())
else:
    print("⚠️ Tidak ada data Twitter untuk diproses")

# ============================================================================
# 7. LOAD DAN PREPROCESSING DATA TIKTOK
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: LOAD DAN PREPROCESSING DATA TIKTOK")
print("=" * 80)

# Load TikTok data
try:
    print("Loading TikTok data...")
    df_tiktok = pd.read_csv(
        DATA_PATH + 'Tiktok-comment.csv',
        header=None,
        encoding='utf-8',
        on_bad_lines='skip'
    )
    print(f"✓ TikTok data loaded: {len(df_tiktok)} baris")
    print(f"Kolom yang tersedia: {df_tiktok.columns.tolist()}")
    
    # Kolom ke-1 (indeks 0) diasumsikan sebagai teks komentar
    df_tiktok = df_tiktok[[0]].rename(columns={0: 'text'})
    df_tiktok['source'] = 'TikTok'
    
    print(f"✓ Menggunakan kolom indeks 0 sebagai 'text'")
    print(f"Preview data TikTok:")
    print(df_tiktok.head())
    
except Exception as e:
    print(f"⚠️ Error loading TikTok data: {e}")
    print("Creating empty TikTok dataframe...")
    df_tiktok = pd.DataFrame({'text': [], 'source': []})

# Preprocessing TikTok
if len(df_tiktok) > 0:
    print("\nMemproses data TikTok...")
    df_tiktok['cleaned_text'] = df_tiktok['text'].apply(clean_text_tiktok)
    df_tiktok = df_tiktok[df_tiktok['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Labeling dengan InSet
    print("Melakukan labeling sentimen...")
    sentiments = []
    scores = []
    for text in df_tiktok['cleaned_text']:
        sentiment, score = inset_sentiment_labeling(text, positive_lexicon, negative_lexicon)
        sentiments.append(sentiment)
        scores.append(score)
    
    df_tiktok['sentiment'] = sentiments
    df_tiktok['sentiment_score'] = scores
    df_tiktok['source'] = 'tiktok'
    
    print(f"✓ TikTok preprocessing selesai!")
    print(f"Data setelah cleaning: {len(df_tiktok)} baris")
    print(f"\nDistribusi sentimen TikTok:")
    print(df_tiktok['sentiment'].value_counts())
else:
    print("⚠️ Tidak ada data TikTok untuk diproses")

# ============================================================================
# 8. KOMBINASI DATA TWITTER DAN TIKTOK
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: KOMBINASI DATA TWITTER DAN TIKTOK")
print("=" * 80)

# Pilih kolom yang relevan
twitter_selected = df_twitter[['cleaned_text', 'sentiment', 'sentiment_score', 'source']].copy()
tiktok_selected = df_tiktok[['cleaned_text', 'sentiment', 'sentiment_score', 'source']].copy()

# Gabungkan data
df_combined = pd.concat([twitter_selected, tiktok_selected], ignore_index=True)

# Shuffle data
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"✓ Data berhasil digabungkan!")
print(f"Total data: {len(df_combined)} baris")
print(f"  - Twitter: {len(twitter_selected)} baris")
print(f"  - TikTok: {len(tiktok_selected)} baris")

print(f"\nDistribusi sentimen gabungan:")
print(df_combined['sentiment'].value_counts())
print(f"\nPersentase:")
print(df_combined['sentiment'].value_counts(normalize=True) * 100)

# Statistik per source
print(f"\nDistribusi per source:")
print(pd.crosstab(df_combined['source'], df_combined['sentiment']))

# ============================================================================
# 9. EXPORT KE CSV BARU
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: EXPORT DATA GABUNGAN KE CSV")
print("=" * 80)

# Export combined data
output_filename = 'coretax_sentiment_combined.csv'
df_combined.to_csv(output_filename, index=False)

print(f"✓ Data berhasil di-export ke: {output_filename}")
print(f"\nPreview data:")
print(df_combined.head(10))

# Download file
from google.colab import files
files.download(output_filename)

print(f"\n✓ File siap didownload!")

# Visualisasi data gabungan
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Distribusi sentimen keseluruhan
sentiment_counts = df_combined['sentiment'].value_counts()
colors = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
bars1 = axes[0].bar(sentiment_counts.index, sentiment_counts.values,
                    color=[colors[s] for s in sentiment_counts.index])
axes[0].set_title('Distribusi Sentimen Gabungan', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sentimen', fontsize=12)
axes[0].set_ylabel('Jumlah', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Distribusi per source
source_sentiment = pd.crosstab(df_combined['source'], df_combined['sentiment'])
source_sentiment.plot(kind='bar', ax=axes[1], color=[colors[s] for s in source_sentiment.columns])
axes[1].set_title('Distribusi Sentimen per Source', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Source', fontsize=12)
axes[1].set_ylabel('Jumlah', fontsize=12)
axes[1].legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].grid(axis='y', alpha=0.3)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# ============================================================================
# 10. SPLITTING DATA UNTUK MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 10: SPLITTING DATA")
print("=" * 80)

# Filter hanya positive dan negative untuk binary classification
df_binary = df_combined[df_combined['sentiment'].isin(['positive', 'negative'])].copy()
df_binary['label'] = (df_binary['sentiment'] == 'positive').astype(int)

print(f"Data untuk training (binary classification): {len(df_binary)} baris")
print(f"  - Positive: {(df_binary['label'] == 1).sum()}")
print(f"  - Negative: {(df_binary['label'] == 0).sum()}")

# Split data: 80% train, 20% test
X = df_binary['cleaned_text'].values
y = df_binary['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data splitting selesai!")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# ============================================================================
# 11. PENERAPAN ALGORITMA - TF-IDF
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: PENERAPAN ALGORITMA - TF-IDF")
print("=" * 80)

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"✓ TF-IDF vectorization selesai!")
print(f"Feature shape: {X_train_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# Train Logistic Regression
print("\nTraining TF-IDF + Logistic Regression...")
tfidf_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
tfidf_model.fit(X_train_tfidf, y_train)

# Predictions
y_pred_tfidf = tfidf_model.predict(X_test_tfidf)

# Evaluation
tfidf_accuracy = accuracy_score(y_test, y_pred_tfidf)
tfidf_f1 = f1_score(y_test, y_pred_tfidf)

print(f"\n✓ TF-IDF Model Training selesai!")
print(f"\nTF-IDF Model Performance:")
print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"F1-Score: {tfidf_f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_tfidf,
                          target_names=['Negative', 'Positive']))

# ============================================================================
# 12. PENERAPAN ALGORITMA - IndoBERT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: PENERAPAN ALGORITMA - IndoBERT")
print("=" * 80)

# Load IndoBERT
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

print(f"✓ IndoBERT model loaded!")

# Dataset class
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

print(f"✓ Datasets created!")

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",  # Updated from evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
print("Training IndoBERT...")
trainer.train()

# Evaluate
predictions = trainer.predict(test_dataset)
y_pred_bert = np.argmax(predictions.predictions, axis=1)

bert_accuracy = accuracy_score(y_test, y_pred_bert)
bert_f1 = f1_score(y_test, y_pred_bert)

print(f"\n✓ IndoBERT Model Training selesai!")
print(f"\nIndoBERT Model Performance:")
print(f"Accuracy: {bert_accuracy:.4f}")
print(f"F1-Score: {bert_f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_bert,
                          target_names=['Negative', 'Positive']))

# ============================================================================
# 13. VISUALISASI PERBANDINGAN MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: VISUALISASI PERBANDINGAN MODEL")
print("=" * 80)

# Comparison
models = ['TF-IDF + LR', 'IndoBERT']
accuracies = [tfidf_accuracy, bert_accuracy]
f1_scores = [tfidf_f1, bert_f1]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Accuracy
axes[0].bar(models, accuracies, color=['#3498db', '#e74c3c'], alpha=0.8)
axes[0].set_title('Perbandingan Accuracy', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_ylim([0, 1])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# F1-Score
axes[1].bar(models, f1_scores, color=['#3498db', '#e74c3c'], alpha=0.8)
axes[1].set_title('Perbandingan F1-Score', fontsize=14, fontweight='bold')
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cm_tfidf = confusion_matrix(y_test, y_pred_tfidf)
sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix - TF-IDF + LR', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

cm_bert = confusion_matrix(y_test, y_pred_bert)
sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Reds',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix - IndoBERT', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.show()

# ============================================================================
# 14. VISUALISASI PERBANDINGAN POSITIF, NEGATIF, DAN NETRAL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 14: VISUALISASI PERBANDINGAN POSITIF, NEGATIF, DAN NETRAL")
print("=" * 80)

# Separate texts untuk semua sentimen (termasuk neutral)
positive_texts = ' '.join(df_combined[df_combined['sentiment'] == 'positive']['cleaned_text'])
negative_texts = ' '.join(df_combined[df_combined['sentiment'] == 'negative']['cleaned_text'])
neutral_texts = ' '.join(df_combined[df_combined['sentiment'] == 'neutral']['cleaned_text'])

print(f"Generating WordClouds...")
print(f"  - Positive texts: {len(positive_texts.split())} words")
print(f"  - Negative texts: {len(negative_texts.split())} words")
print(f"  - Neutral texts: {len(neutral_texts.split())} words")

# WordClouds untuk 3 sentimen
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Positive WordCloud
if len(positive_texts.strip()) > 0:
    wordcloud_pos = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap='Greens',
        max_words=100
    ).generate(positive_texts)
    
    axes[0].imshow(wordcloud_pos, interpolation='bilinear')
    axes[0].set_title('WordCloud - Sentimen POSITIF', fontsize=16, fontweight='bold', color='green')
    axes[0].axis('off')
else:
    axes[0].text(0.5, 0.5, 'Tidak ada data positif', ha='center', va='center', fontsize=14)
    axes[0].axis('off')

# Negative WordCloud
if len(negative_texts.strip()) > 0:
    wordcloud_neg = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap='Reds',
        max_words=100
    ).generate(negative_texts)
    
    axes[1].imshow(wordcloud_neg, interpolation='bilinear')
    axes[1].set_title('WordCloud - Sentimen NEGATIF', fontsize=16, fontweight='bold', color='red')
    axes[1].axis('off')
else:
    axes[1].text(0.5, 0.5, 'Tidak ada data negatif', ha='center', va='center', fontsize=14)
    axes[1].axis('off')

# Neutral WordCloud
if len(neutral_texts.strip()) > 0:
    wordcloud_neu = WordCloud(
        width=600,
        height=400,
        background_color='white',
        colormap='Greys',
        max_words=100
    ).generate(neutral_texts)
    
    axes[2].imshow(wordcloud_neu, interpolation='bilinear')
    axes[2].set_title('WordCloud - Sentimen NETRAL', fontsize=16, fontweight='bold', color='gray')
    axes[2].axis('off')
else:
    axes[2].text(0.5, 0.5, 'Tidak ada data netral', ha='center', va='center', fontsize=14)
    axes[2].axis('off')

plt.tight_layout()
plt.show()

# Top words untuk semua sentimen
from collections import Counter

def get_top_words(text, n=15):
    if not text or len(text.strip()) == 0:
        return []
    words = text.split()
    words = [w for w in words if len(w) > 3]
    return Counter(words).most_common(n)

top_positive = get_top_words(positive_texts, 15)
top_negative = get_top_words(negative_texts, 15)
top_neutral = get_top_words(neutral_texts, 15)

# Plot top words untuk 3 sentimen
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# Top Positive Words
if top_positive:
    pos_words, pos_counts = zip(*top_positive)
    axes[0].barh(pos_words, pos_counts, color='#2ecc71', alpha=0.8)
    axes[0].set_title('Top 15 Kata - Sentimen POSITIF', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Frekuensi', fontsize=12)
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
else:
    axes[0].text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', fontsize=12)
    axes[0].set_title('Top 15 Kata - Sentimen POSITIF', fontsize=14, fontweight='bold')

# Top Negative Words
if top_negative:
    neg_words, neg_counts = zip(*top_negative)
    axes[1].barh(neg_words, neg_counts, color='#e74c3c', alpha=0.8)
    axes[1].set_title('Top 15 Kata - Sentimen NEGATIF', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Frekuensi', fontsize=12)
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', fontsize=12)
    axes[1].set_title('Top 15 Kata - Sentimen NEGATIF', fontsize=14, fontweight='bold')

# Top Neutral Words
if top_neutral:
    neu_words, neu_counts = zip(*top_neutral)
    axes[2].barh(neu_words, neu_counts, color='#95a5a6', alpha=0.8)
    axes[2].set_title('Top 15 Kata - Sentimen NETRAL', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Frekuensi', fontsize=12)
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3)
else:
    axes[2].text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', fontsize=12)
    axes[2].set_title('Top 15 Kata - Sentimen NETRAL', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 15. VISUALISASI PIE CHART DISTRIBUSI SENTIMEN
# ============================================================================
print("\n" + "=" * 80)
print("STEP 15: VISUALISASI PIE CHART DISTRIBUSI SENTIMEN")
print("=" * 80)

# Pie chart untuk distribusi sentimen keseluruhan
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# 1. Pie Chart - Distribusi Sentimen Keseluruhan
sentiment_counts = df_combined['sentiment'].value_counts()
colors_pie = {'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'}
pie_colors = [colors_pie.get(s, '#95a5a6') for s in sentiment_counts.index]

explode = [0.05] * len(sentiment_counts)  # Sedikit pisahkan semua slice

axes[0, 0].pie(sentiment_counts.values, 
               labels=sentiment_counts.index, 
               autopct='%1.1f%%',
               colors=pie_colors,
               explode=explode,
               shadow=True,
               startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0, 0].set_title('Distribusi Sentimen Keseluruhan\n(Twitter + TikTok)', 
                     fontsize=14, fontweight='bold', pad=20)

# 2. Pie Chart - Distribusi Sentimen Twitter
twitter_sentiment = df_combined[df_combined['source'] == 'twitter']['sentiment'].value_counts()
twitter_colors = [colors_pie.get(s, '#95a5a6') for s in twitter_sentiment.index]
twitter_explode = [0.05] * len(twitter_sentiment)

axes[0, 1].pie(twitter_sentiment.values,
               labels=twitter_sentiment.index,
               autopct='%1.1f%%',
               colors=twitter_colors,
               explode=twitter_explode,
               shadow=True,
               startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0, 1].set_title('Distribusi Sentimen Twitter', 
                     fontsize=14, fontweight='bold', pad=20)

# 3. Pie Chart - Distribusi Sentimen TikTok
tiktok_sentiment = df_combined[df_combined['source'] == 'tiktok']['sentiment'].value_counts()
tiktok_colors = [colors_pie.get(s, '#95a5a6') for s in tiktok_sentiment.index]
tiktok_explode = [0.05] * len(tiktok_sentiment)

axes[1, 0].pie(tiktok_sentiment.values,
               labels=tiktok_sentiment.index,
               autopct='%1.1f%%',
               colors=tiktok_colors,
               explode=tiktok_explode,
               shadow=True,
               startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1, 0].set_title('Distribusi Sentimen TikTok', 
                     fontsize=14, fontweight='bold', pad=20)

# 4. Pie Chart - Perbandingan Source (Twitter vs TikTok)
source_counts = df_combined['source'].value_counts()
source_colors = ['#1DA1F2', '#000000']  # Twitter blue, TikTok black

axes[1, 1].pie(source_counts.values,
               labels=['Twitter', 'TikTok'],
               autopct='%1.1f%%',
               colors=source_colors,
               explode=[0.05, 0.05],
               shadow=True,
               startangle=90,
               textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'})
axes[1, 1].set_title('Perbandingan Jumlah Data\n(Twitter vs TikTok)', 
                     fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# Print statistik detail
print("\n📊 STATISTIK DETAIL SENTIMEN:")
print("\n1. Keseluruhan:")
for sentiment in ['positive', 'negative', 'neutral']:
    count = (df_combined['sentiment'] == sentiment).sum()
    percentage = (count / len(df_combined)) * 100
    print(f"   {sentiment.capitalize()}: {count} ({percentage:.2f}%)")

print("\n2. Twitter:")
twitter_data = df_combined[df_combined['source'] == 'twitter']
for sentiment in ['positive', 'negative', 'neutral']:
    count = (twitter_data['sentiment'] == sentiment).sum()
    percentage = (count / len(twitter_data)) * 100 if len(twitter_data) > 0 else 0
    print(f"   {sentiment.capitalize()}: {count} ({percentage:.2f}%)")

print("\n3. TikTok:")
tiktok_data = df_combined[df_combined['source'] == 'tiktok']
for sentiment in ['positive', 'negative', 'neutral']:
    count = (tiktok_data['sentiment'] == sentiment).sum()
    percentage = (count / len(tiktok_data)) * 100 if len(tiktok_data) > 0 else 0
    print(f"   {sentiment.capitalize()}: {count} ({percentage:.2f}%)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY - SENTIMENT ANALYSIS SELESAI!")
print("=" * 80)

print(f"\n📊 DATA STATISTICS:")
print(f"Total data gabungan: {len(df_combined)} baris")
print(f"  - Twitter: {len(twitter_selected)} baris")
print(f"  - TikTok: {len(tiktok_selected)} baris")
print(f"Data untuk training: {len(df_binary)} baris")
print(f"  - Training set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")

print(f"\n📈 MODEL PERFORMANCE:")
print(f"\nTF-IDF + Logistic Regression:")
print(f"  - Accuracy: {tfidf_accuracy:.4f}")
print(f"  - F1-Score: {tfidf_f1:.4f}")

print(f"\nIndoBERT:")
print(f"  - Accuracy: {bert_accuracy:.4f}")
print(f"  - F1-Score: {bert_f1:.4f}")

print(f"\n🏆 BEST MODEL: ", end="")
if bert_accuracy > tfidf_accuracy:
    print("IndoBERT")
    print(f"   (Accuracy lebih tinggi: {bert_accuracy:.4f} vs {tfidf_accuracy:.4f})")
else:
    print("TF-IDF + Logistic Regression")
    print(f"   (Accuracy lebih tinggi: {tfidf_accuracy:.4f} vs {bert_accuracy:.4f})")

print(f"\n📁 OUTPUT FILE: {output_filename}")
print("\n✅ Semua tahapan selesai!")
print("=" * 80)
