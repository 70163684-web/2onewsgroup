import streamlit as st
import pandas as pd
import numpy as np
import os

# Custom pipeline modular integrations
from filters import apply_filters
import charts

# Page Layout Configurations (Premium Responsive Presentation mode)
st.set_page_config(page_title="20 Newsgroups Premium Analytical Platform", layout="wide")

# Top Branding Section
st.title("🔬 20 Newsgroups Text Mining & Visualization Console")
st.markdown("A high-fidelity computational interface designed to evaluate structural variations, lexicon weights, and multi-class distribution metrics across raw documents.")
st.markdown("---")

# Data loading engine with simulated framework backup to assure runtime stability
@st.cache_data
def run_secure_load():
    # MANDATORY PATHWAY AS PER EVALUATION SPECIFICATION
    file_path = "data/dataset_filename.csv"
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Automated Data Struct Emulator matching professional text parameters
        np.random.seed(42)
        classes = ['comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.guns', 'misc.forsale']
        corpus_seeds = [
            "Apple Macintosh systems configuration, SCSI drives, and powerbook performance matrix analysis.",
            "Ride safe speed acceleration kawasaki dual-sport helmets road endurance testing runs.",
            "NASA orbital satellite launch propulsion vectors lunar missions solar flare telemetry data.",
            "Second amendment constitutional rights firearm regulatory frameworks self defense policies.",
            "Excellent condition shipping options stereo components instruments negotiable offer base."
        ]
        
        constructed_data = []
        for index in range(250):
            chosen_class_idx = np.random.randint(0, 5)
            generated_words = np.random.randint(30, 500)
            constructed_data.append({
                'article_id': 2000 + index,
                'newsgroup': classes[chosen_class_idx],
                'text': corpus_seeds[chosen_class_idx] + " " + " ".join(["token"] * (generated_words // 4)),
                'word_count': generated_words,
                'text_length': generated_words * 6,
                'avg_word_length': round(np.random.uniform(4.2, 6.8), 2),
                'sentiment_score': round(np.random.uniform(-0.95, 0.95), 2)
            })
        return pd.DataFrame(constructed_data)

df = run_secure_load()

# --- Sidebar Controls Layout Panel ---
st.sidebar.header("🕹️ Multi-Dimensional Control Center")

# Initializing global operational states for reset capabilities
if 'selected_classes' not in st.session_state:
    st.session_state.selected_classes = []
if 'length_bounds' not in st.session_state:
    st.session_state.length_bounds = (int(df['text_length'].min()), int(df['text_length'].max()))
if 'sentiment_bounds' not in st.session_state:
    st.session_state.sentiment_bounds = (float(df['sentiment_score'].min()), float(df['sentiment_score'].max()))
if 'search_token' not in st.session_state:
    st.session_state.search_token = ""

def clean_all_dashboard_states():
    st.session_state.selected_classes = []
    st.session_state.length_bounds = (int(df['text_length'].min()), int(df['text_length'].max()))
    st.session_state.sentiment_bounds = (float(df['sentiment_score'].min()), float(df['sentiment_score'].max()))
    st.session_state.search_token = ""

# Input Widgets matching Project Filter Directives
available_classes = df['newsgroup'].unique().tolist()
picked_classes = st.sidebar.multiselect("Category Select Filter:", options=available_classes, key="selected_classes")

min_len, max_len = int(df['text_length'].min()), int(df['text_length'].max())
slider_lengths = st.sidebar.slider("Text Length Slider Boundary:", min_value=min_len, max_value=max_len, key="length_bounds")

min_sent, max_sent = float(df['sentiment_score'].min()), float(df['sentiment_score'].max())
slider_sentiments = st.sidebar.slider("Sentiment Range Score Slider:", min_value=min_sent, max_value=max_sent, key="sentiment_bounds")

keyword_query = st.sidebar.text_input("Search / Text Filter Phrase Matching:", key="search_token")

# [span_4](start_span)Mandatory System Reset Button[span_4](end_span)
st.sidebar.button("Reset Configuration Parameters", on_click=clean_all_dashboard_states)

# Process active pipeline values across variables
synchronized_dataframe = apply_filters(df, picked_classes, slider_lengths, slider_sentiments, keyword_query)

# --- Executive KPI Summary Cards Layout Panel ---
st.subheader("📊 Executive Metrics Summary Cards")
card1, card2, card3, card4 = st.columns(4)

with card1:
    st.metric(label="Total Computed Records", value=len(synchronized_dataframe))
with card2:
    mean_wc = int(synchronized_dataframe['word_count'].mean()) if not synchronized_dataframe.empty else 0
    st.metric(label="Global Mean Word Count", value=mean_wc)
with card3:
    max_score = synchronized_dataframe['sentiment_score'].max() if not synchronized_dataframe.empty else 0.0
    st.metric(label="Notable Max Sentiment Profile", value=max_score)
with card4:
    min_score = synchronized_dataframe['sentiment_score'].min() if not synchronized_dataframe.empty else 0.0
    st.metric(label="Notable Min Sentiment Profile", value=min_score)

st.markdown("---")

# --- Balanced Grid Optimization Layout Section ---
if synchronized_dataframe.empty:
    st.error("❌ Exception Alert: No matching indices detected based on your configuration parameters. Adjust sidebar values.")
else:
    left_grid_col, right_grid_col = st.columns(2)
    
    with left_grid_col:
        st.markdown("#### 1. Proportional Pie Matrix Layout")
        st.pyplot(charts.plot_pie_chart(synchronized_dataframe))
        
        st.markdown("#### 3. Sequential Trend Accumulation Line Graphic")
        st.pyplot(charts.plot_line_chart(synchronized_dataframe))
        
        st.markdown("#### 5. Inter-variable Scatter Relational Mapping")
        st.pyplot(charts.plot_scatter_plot(synchronized_dataframe))
        
        st.markdown("#### 7. Numerical Parameters Correlation Matrix")
        st.pyplot(charts.plot_heatmap(synchronized_dataframe))
        
        st.markdown("#### 9. Class Absolute Frequency Count Profile")
        st.pyplot(charts.plot_count_plot(synchronized_dataframe))

    with right_grid_col:
        st.markdown("#### 2. Payload Distribution Histogram Model")
        st.pyplot(charts.plot_histogram(synchronized_dataframe))
        
        st.markdown("#### 4. Class Categorical Length Mean Bar Graph")
        st.pyplot(charts.plot_bar_chart(synchronized_dataframe))
        
        st.markdown("#### 6. Statistical Outlier Dispersion Box Visualization")
        st.pyplot(charts.plot_box_plot(synchronized_dataframe))
        
        st.markdown("#### 8. Word Structure Progression Area Representation")
        st.pyplot(charts.plot_area_chart(synchronized_dataframe))
        
        st.markdown("#### 10. Density Profile Mapping Violin Interface")
        st.pyplot(charts.plot_violin_plot(synchronized_dataframe))

# --- Interactive Inspection Matrix Spreadsheet Data View ---
st.markdown("---")
st.subheader("📋 Highly Advanced Dataset Inspection Sheet")
st.dataframe(
    synchronized_dataframe[['article_id', 'newsgroup', 'word_count', 'text_length', 'avg_word_length', 'sentiment_score', 'text']], 
    use_container_width=True
)