import streamlit as st
from utils import load_data, setup_page
import plotly.express as px

def show():
    setup_page("Analysis")

    st.title("Analisis Mendalam")
    df = load_data()
    
    if df.empty:
        st.error("Data tidak ditemukan. Harap pastikan file CSV tersedia.")
        return

    tab1, tab2, tab3 = st.tabs(["Pain Points (Negatif)", "Success Stories (Positif)", "Eksplorasi Topik"])

    with tab1:
        st.header("Identifikasi Masalah (Pain Points)")
        st.markdown("Analisis ini berfokus pada komentar bersentimen **Negatif** untuk menemukan kendala utama pengguna.")
        
        # Filter Negative
        neg_df = df[df['sentiment'] == 'negative']
        
        if 'topic_name' in neg_df.columns:
            topic_counts = neg_df['topic_name'].value_counts().head(10).reset_index()
            topic_counts.columns = ['Topik', 'Jumlah']
            # Clean topic names
            topic_counts['Topik'] = topic_counts['Topik'].apply(lambda x: " ".join(x.split('_')[1:]) if isinstance(x, str) else x)
            
            fig = px.bar(topic_counts, x='Jumlah', y='Topik', orientation='h', title="Topik Keluhan Utama", color='Jumlah', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Insight Utama")
            st.info("Grafik di atas menunjukkan topik-topik yang paling sering muncul dalam komentar negatif. Ini adalah area prioritas untuk perbaikan.")
        else:
            st.warning("Kolom topik tidak ditemukan dalam data.")

        st.subheader("Analisis Kata Kunci (Word Cloud)")
        try:
            st.image("outputs/wordcloud-sentiment.png", caption="Word Cloud Sentimen Negatif", use_column_width=True)
        except:
            st.warning("Gambar Word Cloud tidak ditemukan.")

    with tab2:
        st.header("Hal Positif (Success Stories)")
        st.markdown("Apa yang disukai pengguna dari CoreTax?")
        
        pos_df = df[df['sentiment'] == 'positive']
        
        if 'topic_name' in pos_df.columns:
            pos_topic_counts = pos_df['topic_name'].value_counts().head(10).reset_index()
            pos_topic_counts.columns = ['Topik', 'Jumlah']
            pos_topic_counts['Topik'] = pos_topic_counts['Topik'].apply(lambda x: " ".join(x.split('_')[1:]) if isinstance(x, str) else x)
            
            fig_pos = px.bar(pos_topic_counts, x='Jumlah', y='Topik', orientation='h', title="Topik Apresiasi Utama", color='Jumlah', color_continuous_scale='Greens')
            st.plotly_chart(fig_pos, use_container_width=True)
        else:
            st.warning("Kolom topik tidak ditemukan.")

    with tab3:
        st.header("Eksplorasi Topik")
        st.markdown("Lihat detail komentar untuk setiap topik.")
        
        if 'topic_name' in df.columns:
            all_topics = sorted(df['topic_name'].unique().tolist())
            selected_topic = st.selectbox("Pilih Topik:", all_topics)
            
            subset = df[df['topic_name'] == selected_topic]
            
            st.metric("Jumlah Komentar", len(subset))
            
            st.subheader("Sampel Komentar")
            for txt in subset['text'].head(5):
                st.markdown(f"> {txt}")
                st.markdown("---")
        else:
            st.warning("Data topik tidak tersedia.")

if __name__ == "__main__":
    show()
