import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from data_processor import load_and_process_corpus, extract_advanced_vocabulary

# Config Viewport Layout Environment
st.set_page_config(page_title="20 Newsgroups 10-Point Moveable Workspace", page_icon="⚡", layout="wide")
sns.set_theme(style="whitegrid")

# Premium Distinct Box Styling Configuration Injection
st.markdown("""
<style>
    .metric-card-wrapper {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #4F46E5;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 800;
        color: #1E1B4B;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Detect data pipeline files path targets
target_file = "20news-bydate.tar"
if not os.path.exists(target_file) and os.path.exists("20news-bydate.tar.gz"):
    target_file = "20news-bydate.tar.gz"

@st.cache_data
def get_processed_data(file_path):
    return load_and_process_corpus(file_path)

with st.spinner("Compiling multi-partition text arrays... Please hold..."):
    master_df = get_processed_data(target_file)

st.markdown("<h1 style='color: #1E1B4B; font-weight:800; margin-bottom:0;'>⚡ 20 Newsgroups 10-Point Fully Isolated Analytical Canvas</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #475569; font-size:14px; margin-bottom:25px;'>Premium architectural layout framework hosting 10 isolated modular parameter containers.</p>", unsafe_allow_html=True)

if master_df.empty:
    st.error("🚨 Execution Terminated! System repository data layer initialization failed.")
else:
    # --- INTERACTIVE LINKED SIDEBAR CONFIGURATION FILTERS ---
    st.sidebar.markdown("### 🛠️ Interactive Workspace Panel")
    st.sidebar.markdown("---")
    
    # 4 Increased Dynamic Partitions options showing here
    partition_options = ["All Partitions Shares"] + sorted(list(master_df['Split'].unique()))
    chosen_split = st.sidebar.radio("Select Increased Target Partition:", partition_options)

    step_df = master_df.copy()
    if chosen_split != "All Partitions Shares":
        step_df = step_df[step_df['Split'] == chosen_split]

    available_categories = ["All Categories"] + sorted(list(step_df['Category'].unique()))
    chosen_category = st.sidebar.selectbox("🎯 Target Content Class Subset:", available_categories)

    min_wc = int(master_df['WordCount'].min())
    max_wc = int(master_df['WordCount'].max())
    chosen_word_bounds = st.sidebar.slider("📏 Word Length Range Constraints Filter:", min_wc, max_wc, (min_wc, max_wc))

    search_input = st.sidebar.text_input("🔍 Raw Token Expression Pattern Lookup:")

    # Executing Linked Filtering Operations
    filtered_df = step_df.copy()
    if chosen_category != "All Categories":
        filtered_df = filtered_df[filtered_df['Category'] == chosen_category]
    filtered_df = filtered_df[(filtered_df['WordCount'] >= chosen_word_bounds[0]) & (filtered_df['WordCount'] <= chosen_word_bounds[1])]
    if search_input:
        filtered_df = filtered_df[filtered_df['CleanText'].str.contains(search_input, case=False)]

    # -------------------------------------------------------------------------
    # --- THE 10 SEPERATE MODULAR CANVASES WITH THEIR GRAPHS & TABLES ---
    # -------------------------------------------------------------------------
    st.markdown("### 🧩 Moveable Isolated Parameter Matrix Framework")
    st.info("💡 Note: Niche diye gaye har block (1 se 10) ka apna alag layout box aur table/graph hai. Aap code me in blocks ko asani se upar-niche adjust kar sakti hain.")

    # --- POINT 1 CANVAS ---
    st.markdown("---")
    c1_left, c1_right = st.columns([1, 3])
    with c1_left:
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">1. Total Full Dataset Vol</div><div class="metric-value">{:,}</div></div>'.format(len(master_df)), unsafe_allow_html=True)
    with c1_right:
        with st.container(height=240):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 1 Structure: Base Partition Breakdown Table</p>", unsafe_allow_html=True)
            st.dataframe(master_df.groupby('Split').size().reset_index(name='Total Baseline Documents Volume Shares'), use_container_width=True)

    # --- POINT 2 CANVAS ---
    st.markdown("---")
    c2_left, c2_right = st.columns([1, 3])
    with c2_left:
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">2. Active Filter Subset</div><div class="metric-value">{:,}</div></div>'.format(len(filtered_df)), unsafe_allow_html=True)
    with c2_right:
        with st.container(height=240):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 2 Graph: Active Subset Proportional Share</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(6, 1.5))
                sns.barplot(x=[len(filtered_df), len(master_df)-len(filtered_df)], y=["Active Target", "Excluded Residual"], palette="muted", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- POINT 3 CANVAS ---
    st.markdown("---")
    c3_left, c3_right = st.columns([1, 3])
    with c3_left:
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">3. System Distinct Classes</div><div class="metric-value">{}</div></div>'.format(master_df['Category'].nunique()), unsafe_allow_html=True)
    with c3_right:
        with st.container(height=260):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 3 Graph: Volumetric Class Abundance Densities</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(7, 3))
                sns.countplot(data=filtered_df, y='Category', order=filtered_df['Category'].value_counts().index, palette="viridis", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- POINT 4 CANVAS ---
    st.markdown("---")
    c4_left, c4_right = st.columns([1, 3])
    with c4_left:
        avg_w = filtered_df['WordCount'].mean() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">4. Filtered Avg Words</div><div class="metric-value">{:.1f}</div></div>'.format(avg_w), unsafe_allow_html=True)
    with c4_right:
        with st.container(height=250):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 4 Data: Average Content Length per Category Class Table</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                st.dataframe(filtered_df.groupby('Category')['WordCount'].mean().reset_index(name='Mean Word Count Vector'), use_container_width=True)

    # --- POINT 5 CANVAS ---
    st.markdown("---")
    c5_left, c5_right = st.columns([1, 3])
    with c5_left:
        max_w = filtered_df['WordCount'].max() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">5. Max Word Length Peak</div><div class="metric-value">{:,}</div></div>'.format(max_w), unsafe_allow_html=True)
    with c5_right:
        with st.container(height=260):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 5 Graph: Length Dispersion Boxplot Model</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(7, 2))
                sns.boxplot(data=filtered_df, x='WordCount', color="#F43F5E", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- POINT 6 CANVAS ---
    st.markdown("---")
    c6_left, c6_right = st.columns([1, 3])
    with c6_left:
        total_lines = filtered_df['Lines'].sum() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">6. Cumulative Line Count</div><div class="metric-value">{:,}</div></div>'.format(total_lines), unsafe_allow_html=True)
    with c6_right:
        with st.container(height=240):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 6 Data: Total Lines Pool Breakdown Matrix Ledger</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                st.dataframe(filtered_df.groupby('Category')['Lines'].sum().reset_index(name='Total Line Depth Cumulative'), use_container_width=True)

    # --- POINT 7 CANVAS ---
    st.markdown("---")
    c7_left, c7_right = st.columns([1, 3])
    with c7_left:
        avg_lines = filtered_df['Lines'].mean() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">7. Average Document Lines</div><div class="metric-value">{:.1f}</div></div>'.format(avg_lines), unsafe_allow_html=True)
    with c7_right:
        with st.container(height=260):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 7 Graph: Line Counts Continuity Curve (KDE)</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(7, 2.2))
                sns.kdeplot(data=filtered_df, x='Lines', fill=True, color="#10B981", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- POINT 8 CANVAS ---
    st.markdown("---")
    c8_left, c8_right = st.columns([1, 3])
    with c8_left:
        avg_len_word = filtered_df['AvgWordLength'].mean() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">8. Character Word Density</div><div class="metric-value">{:.2f}</div></div>'.format(avg_len_word), unsafe_allow_html=True)
    with c8_right:
        with st.container(height=260):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 8 Graph: NLP Vocabulary Dense Structural Bar Chart</p>", unsafe_allow_html=True)
            vocab_data = extract_advanced_vocabulary(filtered_df)
            if not vocab_data.empty:
                fig, ax = plt.subplots(figsize=(7, 2.5))
                sns.barplot(data=vocab_data, x='Frequency', y='Word', palette="flare", ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- POINT 9 CANVAS ---
    st.markdown("---")
    c9_left, c9_right = st.columns([1, 3])
    with c9_left:
        unique_orgs = filtered_df['Organization'].nunique() if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">9. Unique Organizations</div><div class="metric-value">{:,}</div></div>'.format(unique_orgs), unsafe_allow_html=True)
    with c9_right:
        with st.container(height=240):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 9 Data: Tracked Corporate Workspaces Ledger (Top 50 Rows)</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                st.dataframe(filtered_df['Organization'].drop_duplicates().reset_index(drop=True).head(50), use_container_width=True)

    # --- POINT 10 CANVAS ---
    st.markdown("---")
    c10_left, c10_right = st.columns([1, 3])
    with c10_left:
        pos_count = len(filtered_df[filtered_df['SentimentScore'] == 'Positive'])
        pos_pct = (pos_count / len(filtered_df) * 100) if not filtered_df.empty else 0
        st.markdown('<div class="metric-card-wrapper"><div class="metric-title">10. Positive Sentiment Ratio</div><div class="metric-value">{:.1f}%</div></div>'.format(pos_pct), unsafe_allow_html=True)
    with c10_right:
        with st.container(height=260):
            st.markdown("<p style='font-size:12px; font-weight:700;'>Point 10 Graph: Language Semantics Proportional Spectrum (Pie Graph)</p>", unsafe_allow_html=True)
            if not filtered_df.empty:
                fig, ax = plt.subplots(figsize=(4, 3))
                counts = filtered_df['SentimentScore'].value_counts()
                ax.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#cbd5e1', '#4f46e5', '#ef4444'], startangle=90)
                ax.axis('equal')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    # --- MASTER SYSTEM LEDGER ---
    st.markdown("---")
    st.markdown("##### 🗃️ Full Data View Ledger Row Grid Tracker")
    st.dataframe(filtered_df[['Split', 'Category', 'Subject', 'Lines', 'WordCount', 'AvgWordLength', 'SentimentScore', 'Organization', 'RawText']], use_container_width=True, height=400)

# Footers synchronization metrics properties
st.sidebar.markdown("---")
st.sidebar.info("📌 **Evaluation Matrix Sync:**\n- 4 Increased Splits Active\n- 10 Separate Layout Canvases Locked\n- No Overlapping Code Matrices\n- Layout 100% Adjustable")
