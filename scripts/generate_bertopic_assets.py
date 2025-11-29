import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import os

# Paths
DATA_PATH = "models/BERTopic-CoreTax-data.csv"
OUTPUT_ASSETS_DIR = "streamlit_app/assets"
MODEL_DIR = "models/bertopic_model"

def generate_assets():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Filter for negative sentiment or all data depending on what we want to model
    # The original script modeled negative sentiment. Let's stick to that for consistency, 
    # OR model everything to allow full exploration. 
    # User asked for "visualize_documents(texts)", usually implies the whole corpus or the subset of interest.
    # Let's model the NEGATIVE sentiment as that's the focus of "Pain Points".
    
    neg_data = df[df['sentiment'] == 'negative']
    texts = neg_data['text'].fillna('').tolist()
    
    print(f"Training BERTopic on {len(texts)} negative comments...")
    
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    topic_model = BERTopic(
        language="indonesian",
        embedding_model=embedding_model,
        n_gram_range=(1, 2),
        min_topic_size=15 # Slightly lower to ensure we get topics
    )
    
    topics, probs = topic_model.fit_transform(texts)
    
    print("Saving model...")
    topic_model.save(MODEL_DIR)
    
    print("Generating visualizations...")
    # 1. Topics
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(os.path.join(OUTPUT_ASSETS_DIR, "topics.html"))
    
    # 2. Barchart
    fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
    fig_barchart.write_html(os.path.join(OUTPUT_ASSETS_DIR, "barchart.html"))
    
    # 3. Hierarchy
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_html(os.path.join(OUTPUT_ASSETS_DIR, "hierarchy.html"))

    # 4. Heatmap
    fig_heatmap = topic_model.visualize_heatmap()
    fig_heatmap.write_html(os.path.join(OUTPUT_ASSETS_DIR, "heatmap.html"))
    
    # 5. Documents (Sample)
    # visualize_documents is heavy. We'll try it on a subset if it's too large, but 2k-3k is fine.
    # We need the embeddings for this.
    print("Generating document visualization (this might take a moment)...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    fig_docs = topic_model.visualize_documents(texts, embeddings=embeddings)
    fig_docs.write_html(os.path.join(OUTPUT_ASSETS_DIR, "documents.html"))
    
    print("Done! Assets saved to", OUTPUT_ASSETS_DIR)

if __name__ == "__main__":
    generate_assets()
