import streamlit as st
import streamlit.components.v1 as components
from utils import load_html_asset, setup_page

def show():
    st.title("Visualisasi Interaktif (BERTopic)")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Topik", "Dokumen", "Hierarki", "Heatmap", "Barchart"])
    
    with tab1:
        st.header("Sebaran Topik (Intertopic Distance Map)")
        with st.expander("📖 Cara Membaca Grafik Ini", expanded=True):
            st.markdown("""
            - **Lingkaran**: Mewakili satu topik. Ukuran lingkaran menunjukkan frekuensi topik tersebut (semakin besar = semakin banyak dibicarakan).
            - **Jarak**: Semakin dekat jarak antar lingkaran, semakin mirip topik tersebut secara semantik.
            - **Interaksi**: Arahkan mouse ke lingkaran untuk melihat kata kunci utama topik tersebut.
            """)
        html = load_html_asset("topics.html")
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.warning("Visualisasi Topik belum digenerate.")

    with tab2:
        st.header("Visualisasi Dokumen (Document Projections)")
        with st.expander("📖 Cara Membaca Grafik Ini", expanded=False):
            st.markdown("""
            - **Titik-titik**: Setiap titik mewakili satu dokumen (komentar/tweet).
            - **Warna**: Menunjukkan kelompok topik dari dokumen tersebut.
            - **Klaster**: Dokumen yang berkumpul berdekatan memiliki isi/konteks yang mirip.
            - **Kegunaan**: Melihat sebaran outlier atau dokumen yang tidak masuk ke topik manapun.
            """)
        html = load_html_asset("documents.html")
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.warning("Visualisasi Dokumen belum digenerate.")
            
    with tab3:
        st.header("Hierarki Topik")
        with st.expander("📖 Cara Membaca Grafik Ini", expanded=False):
            st.markdown("""
            - **Pohon (Dendrogram)**: Menunjukkan bagaimana topik-topik kecil dapat digabungkan menjadi tema yang lebih besar.
            - **Pengelompokan**: Garis yang menghubungkan topik menunjukkan bahwa mereka memiliki kemiripan yang tinggi dan bisa dianggap sebagai satu kategori besar.
            """)
        html = load_html_asset("hierarchy.html")
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.warning("Visualisasi Hierarki belum digenerate.")

    with tab4:
        st.header("Similarity Heatmap")
        with st.expander("📖 Cara Membaca Grafik Ini", expanded=False):
            st.markdown("""
            - **Matriks Warna**: Menunjukkan skor kemiripan (similarity score) antar topik.
            - **Warna Gelap**: Menunjukkan kemiripan yang tinggi (mendekati 1.0).
            - **Warna Terang**: Menunjukkan topik yang sangat berbeda.
            - **Kegunaan**: Memvalidasi apakah topik-topik yang dihasilkan sudah cukup unik atau ada yang tumpang tindih.
            """)
        html = load_html_asset("heatmap.html")
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.warning("Visualisasi Heatmap belum digenerate.")

    with tab5:
        st.header("Topik per Dokumen (Barchart)")
        with st.expander("📖 Cara Membaca Grafik Ini", expanded=False):
            st.markdown("""
            - **Bar Horizontal**: Menunjukkan skor c-TF-IDF untuk setiap kata dalam topik.
            - **Skor Tinggi**: Kata tersebut sangat penting dan menjadi ciri khas utama dari topik tersebut.
            - **Kegunaan**: Memahami esensi dari setiap topik berdasarkan kata-kata penyusunnya.
            """)
        html = load_html_asset("barchart.html")
        if html:
            components.html(html, height=1200, scrolling=False)
        else:
            st.warning("Visualisasi Barchart belum digenerate.")

if __name__ == "__main__":
    setup_page("Visualizations")
    show()
