import streamlit as st
import pandas as pd
import numpy as np
import os
import tarfile
import re
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG & PREMIUM DARK THEME CODES
# ==========================================
st.set_page_config(
    page_title="Revenue & Psychological Intelligence", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS to mimic the premium video dashboard layout
st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; }
    .metric-box {
        background-color: #1f242c;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-val { color: #58a6ff; font-size: 28px; font-weight: bold; }
    .metric-lbl { color: #8b949e; font-size: 14px; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA LOADING ENGINE (REAL + AUTO-SIMULATION)
# ==========================================
@st.cache_data
def load_and_clean_dataset():
    tar_path = "20news-bydate.tar.gz"
    alternative_tar = "20news-bydate.tar"
    
    category_mapping = {
        'sci.space': 'Academic & Competitive Pressure',
        'comp.sys.mac.hardware': 'Digital Dysmorphia & Cyber Fatigue',
        'rec.motorcycles': 'Corporate Burnout & Grind Culture',
        'talk.politics.guns': 'Panic Attacks & Acute Trauma Triggers',
        'misc.forsale': 'Financial Anxiety & Economic Crisis',
        'alt.atheism': 'Family Expectations & Social Stigma',
        'soc.religion.christian': 'Urban Loneliness & Metro Isolation',
        'sci.med': 'Relationship Friction & Heartbreak'
    }
    
    target_file = tar_path if os.path.exists(tar_path) else (alternative_tar if os.path.exists(alternative_tar) else None)
        
    if target_file is not None:
        all_data = []
        try:
            mode = "r:gz" if target_file.endswith(".gz") else "r"
            with tarfile.open(target_file, mode) as tar:
                for member in tar.getmembers():
                    if member.isfile() and len(member.name.split('/')) >= 3:
                        path_parts = member.name.split('/')
                        raw_newsgroup = path_parts[1]
                        file_id = path_parts[2]
                        newsgroup = category_mapping.get(raw_newsgroup, 'General Psychological Stress')
                        
                        f = tar.extractfile(member)
                        if f is not None:
                            content = f.read().decode('utf-8', errors='ignore')
                            cleaned_content = re.sub(r'\s+', ' ', content).strip()
                            word_count = len(cleaned_content.split())
                            text_length = len(cleaned_content)
                            
                            if word_count > 5:
                                # VADER-like basic token search for real dynamic mapping
                                pos_words = {'good', 'science', 'computer', 'space', 'excellent', 'god', 'happy', 'love', 'heal'}
                                neg_words = {'bad', 'error', 'fail', 'war', 'kill', 'gun', 'wrong', 'problem', 'anxiety', 'stress', 'hurt'}
                                words_set = set(cleaned_content.lower().split())
                                pos_c = len(words_set.intersection(pos_words))
                                neg_c = len(words_set.intersection(neg_words))
                                
                                sentiment = (pos_c - neg_c) / (pos_c + neg_c + 1)
                                sentiment = max(-1.0, min(1.0, sentiment))
                                
                                if sentiment <= -0.4: cat = "Critical / Severely Distressed"
                                elif sentiment <= -0.1: cat = "Mildly Negative"
                                elif sentiment <= 0.2: cat = "Neutral / Observational"
                                elif sentiment <= 0.5: cat = "Seeking Hope / Optimistic"
                                else: cat = "Positive Recovery Status"
                                
                                all_data.append({
                                    'article_id': int(file_id) if file_id.isdigit() else np.random.randint(1000, 5000),
                                    'newsgroup': newsgroup,
                                    'text': cleaned_content[:300] + "...", 
                                    'word_count': word_count,
                                    'text_length': text_length,
                                    'avg_word_length': round(text_length / word_count, 2) if word_count > 0 else 0,
                                    'sentiment_score': round(sentiment, 2),
                                    'sentiment_category': cat,
                                    'simulated_revenue': round(word_count * np.random.uniform(10.5, 45.0), 2)
                                })
            if all_data: return pd.DataFrame(all_data)
        except Exception: pass

    # Fallback/Safety dataset engine matching video complexity scales
    np.random.seed(42)
    classes = list(category_mapping.values())
    sentiment_cats = ["Critical / Severely Distressed", "Mildly Negative", "Neutral / Observational", "Seeking Hope / Optimistic", "Positive Recovery Status"]
    constructed_data = []
    for index in range(1200):
        generated_words = np.random.randint(40, 750)
        sim_sentiment = np.random.uniform(-1.0, 1.0)
        constructed_data.append({
            'article_id': 5000 + index,
            'newsgroup': np.random.choice(classes),
            'text': "Strategic diagnostic text profiling operational anxiety models under commercial workloads.",
            'word_count': generated_words,
            'text_length': generated_words * 5,
            'avg_word_length': round(np.random.uniform(4.2, 6.8), 2),
            'sentiment_score': round(sim_sentiment, 2),
