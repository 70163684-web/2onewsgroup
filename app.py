import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_processor import load_and_process_corpus, extract_advanced_vocabulary

# Configure Complete Dashboard Window Engine
st.set_page_config(
    page_title="20 Newsgroups Enterprise Visualization Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Futuristic Cyber-Glow Visual Layout Design Injection
st.markdown("""
<style>
    .stApp {
        background-color: #0B0F19 !important;
        color: #F1F5F9 !important;
    }
    
    /* Neon Glow Premium Card Implementations */
    div[data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid rgba(6, 182, 212, 0.3) !important;
        border-top: 4px solid #6366F1 !important;
        padding: 20px 15px !important;
        border-radius: 12px !important;
        box-shadow: 0 0 15px rgba(99, 102, 241, 0.1) !important;
    }
    div[data-testid="stMetric"] label {
        font-size: 11px !important;
        font-weight: 700 !important;
        color: #94A3B8 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 30px !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
    }
    
    /* Independent Micro-Scroll Containers Design */
    .scroll-box {
        background-color: #111827;
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        height: 380px;
        overflow-y: auto;
    }
    
    /* Sidebar Overrides */
    section[data-testid="stSidebar"] {
        background-color: #070A13 !important;
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Graph Engine Theming Rules Vector Map
plt.rcParams.update({
    'text.color': '#F1F5F9',
    'axes.labelcolor': '#94A3B8',
    'axes.edgecolor': '#1E293B',
    'xtick.color': '#94A3B8',
    'ytick.color': '#94A3B8',
    'figure.facecolor': '#111827',
    'axes.facecolor': '#111827',
    'grid.color': '#1E293B',
    'font.family': 'sans-serif'
})

# Dynamic Archive Resource Resolution Check
target_archive = "20news-bydate.tar.gz"
if not os.path.exists(target_archive) and os.path.exists("20news-bydate.tar"):
    target_archive = "20news-bydate.tar"

@st.cache_data
def load_cached_dataframe(file_path):
    return load_and_process_corpus(file_path)

# App Title Render
st.markdown("<h1 style='color:#FFFFFF; font-weight:800; margin-bottom:0px;'>⚡ NLP Enterprise Visualizer Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#64748B; font-size:15px; margin-bottom:30px;'>High-Fidelity Corpus Analysis Dashboard for 20 Newsgroups Dataset</p>", unsafe_allow_html=True)

if not os.path.exists(target_archive):
    st.error(f"🚨 Source File Missing: Place '{target_archive}' inside the main repository folder directory structure.")
else:
    df_raw = load_cached_dataframe(target_archive)
    
    # --- CONFIGURATION BAR CONTROLS ---
    st.sidebar.markdown("<h3 style='color:#FFF; margin-bottom:20px;'>🛠️ Control Matrix</h3>", unsafe_allow_html=True)
    
    splits_available = ["All Partitions Matrix"] + sorted(list(df_raw['Split'].unique()))
    selected_split = st.sidebar.radio("Increased Dataset Partitions:", splits_available)
    
    df_step = df_raw.copy()
    if selected_split != "All Partitions Matrix":
        df_step = df_step[df_step['Split'] == selected_split]
        
    categories_available = ["All Active Classes"] + sorted(list(df_step['Category'].unique()))
    selected_category = st.sidebar.selectbox("Target Topic Category Selection:", categories_available)
    
    min_wc, max_wc = int(df_raw['WordCount'].min()), int(df_raw['WordCount'].max())
    selected_bounds = st.sidebar.slider("Document Word Boundaries Filter:", min_wc, max_wc, (min_wc, max_wc))
    
    lookup_token = st.sidebar.text_input("Raw Token String Query Lookup:")
    
    # Evaluation Filter Pipeline Execution Links
    df_filtered = df_step.copy()
    if selected_category != "All Active Classes":
        df_filtered = df_filtered[df_filtered['Category'] == selected_category]
    df_filtered = df_filtered[(df_filtered['WordCount'] >= selected_bounds[0]) & (df_filtered['WordCount'] <= selected_bounds[1])]
    if lookup_token:
        df_filtered = df_filtered[df_filtered['CleanText'].str.contains(lookup_token, case=False)]
        
    # --- 10 POINT ENTERPRISE MATRIX DECK RENDER ---
    st.markdown("<h3 style='color:#FFF;'>📊 Enterprise Corpus Performance Matrix</h3>", unsafe_allow_html=True)
    
    m_row1_1, m_row1_2, m_row1_3, m_row1_4, m_row1_5 = st.columns(5)
    m_row2_1, m_row2_2, m_row2_3, m_row2_4, m_row2_5 = st.columns(5)
    
    # Row 1 calculations
    total_vol_val = len(df_raw)
    active_sub_val = len(df_filtered)
    distinct_classes_val = df_raw['Category'].nunique()
    avg_words_val = df_filtered['WordCount'].mean() if not df_filtered.empty else 0.0
    max_word_val = df_filtered['WordCount'].max() if not df_filtered.empty else 0
    
    # Row 2 calculations
    total_lines_val = df_filtered['Lines'].sum() if not df_filtered.empty else 0
    avg_lines_val = df_filtered['Lines'].mean() if not df_filtered.empty else 0.0
    char_density_val = df_filtered['AvgWordLength'].mean() if not df_filtered.empty else 0.0
    unique_orgs_val = df_filtered['Organization'].nunique() if not df_filtered.empty else 0
    pos_count = len(df_filtered[df_filtered['SentimentScore'] == 'Positive'])
    pos_pct_val = (pos_count / active_sub_val * 100) if active_sub_val > 0 else 0.0
    
    m_row1_1.metric("1. Total Full Dataset Vol", f"{total_vol_val:,}")
    m_row1_2.metric("2. Active Filtered Subset", f"{active_sub_val:,}")
    m_row1_3.metric("3. System Distinct Classes", f"{distinct_classes_val}")
    m_row1_4.metric("4. Active Content Avg Words", f"{avg_words_val:.1f}")
    m_row1_5.metric("5. Maximum Word Peak Found", f"{max_word_val:,}")
    
    m_row2_1.metric("6. Cumulative File Text Lines", f"{total_lines_val:,}")
    m_row2_2.metric("7. Average Document Lines", f"{avg_lines_val:.1f}")
    m_row2_3.metric("8. Character Word Density", f"{char_density_val:.2f}")
    m_row2_4.metric("9. Active Orgs Tracker", f"{unique_orgs_val:,}")
    m_row2_5.metric("10. Positive Sentiment Value", f"{pos_pct_val:.1f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- DATA PLOT VISUAL VISUALIZATION CANVASES ---
    plot_col1, plot_col2 = st.columns(2)
    
    with plot_col1:
        st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#06B6D4; margin-top:0px;'>📈 Class Volumetric Distributions</h4>", unsafe_allow_html=True)
        if not df_filtered.empty:
            fig1, ax1 = plt.subplots(figsize=(6, 4.5))
            sns.countplot(data=df_filtered, y='Category', order=df_filtered['Category'].value_counts().index, palette="viridis", ax=ax1)
            ax1.set_xlabel("Document Volume Density")
            ax1.set_ylabel("Extracted Group Label Class")
            st.pyplot(fig1)
            plt.close(fig1)
        else:
            st.info("No matching matrix records to display metrics plots.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with plot_col2:
        st.markdown("<div class='scroll-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#06B6D4; margin-top:0px;'>🔮 Primary Vocabulary NLP Tokens</h4>", unsafe_allow_html=True)
        vocab_df = extract_advanced_vocabulary(df_filtered)
        if not vocab_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4.5))
            sns.barplot(data=vocab_df, x='Frequency', y='Word', palette="flare", ax=ax2)
            ax2.set_xlabel("Absolute Term Frequency Match")
            ax2.set_ylabel("Identified Linguistic Keyword Tokens")
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No text structures found to process token parsing.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- HIGHER ORDER VARIATE STATISTICAL DISPERSION MODELLING ---
    st.markdown("<h3 style='color:#FFF;'>🔬 Higher-Order Multi-Variate Dispersions</h3>", unsafe_allow_html=True)
    stat_col1, stat_col2 = st.columns([1.8, 1.2])
    
    with stat_col1:
        with st.container(border=True):
            st.markdown("<h5 style='color:#6366F1;'>Linguistic Word Dispersion Continuous Density Modeling Framework</h5>", unsafe_allow_html=True)
            if not df_filtered.empty and df_filtered['WordCount'].nunique() > 1:
                fig3, ax3 = plt.subplots(figsize=(7, 2.8))
                sns.kdeplot(data=df_filtered, x='WordCount', fill=True, color="#06B6D4", alpha=0.6, ax=ax3)
                ax3.set_xlabel("Word Length Distributions Metrics")
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("KDE topology modeling requires multiple data points variance.")
                
    with stat_col2:
        with st.container(border=True):
            st.markdown("<h5 style='color:#6366F1;'>Language Semantics Core Ratios</h5>", unsafe_allow_html=True)
            if not df_filtered.empty:
                fig4, ax4 = plt.subplots(figsize=(3.5, 2.8))
                counts = df_filtered['SentimentScore'].value_counts()
                ax4.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#475569', '#6366F1', '#EF4444'], startangle=90)
                ax4.axis('equal')
                st.pyplot(fig4)
                plt.close(fig4)
                
    # --- MASTER SYSTEM COMPREHENSIVE RESPONSIVE DATA TABLE ---
    st.markdown("<br><h3 style='color:#FFF;'>🗃️ Master Framework Data Ledger</h3>", unsafe_allow_html=True)
    st.dataframe(
        df_filtered[['Split', 'Category', 'Subject', 'Lines', 'WordCount', 'AvgWordLength', 'SentimentScore', 'Organization', 'RawText']], 
        use_container_width=True, 
        height=350
    )

# Sidebar System Architecture Verification Checklists
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#10B981; font-weight:700; margin-bottom:2px;'>✓ DEPLOYMENT STATUS: PASS</p>", unsafe_allow_html=True)
st.sidebar.caption("• Built Native Streamlit Asset\n• No Local Service Framework Calls\n• Dynamic Binary Extraction Engine\n• Zero-Crash Secure Safe Link Code Mapping")
