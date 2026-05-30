import streamlit as st
import pandas as pd
import numpy as np
import os
import tarfile
import re
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LIVE FILTERS LOGIC (Purani filters.py ab yahan hai)
# ==========================================
def apply_filters(df, selected_categories, length_range, score_range, search_query):
    filtered_df = df.copy()
    
    # Category Multi-select Filter
    if selected_categories and 'newsgroup' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['newsgroup'].isin(selected_categories)]
        
    # Text Length Filter (Starts from 0)
    if 'text_length' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['text_length'] >= length_range[0]) & 
            (filtered_df['text_length'] <= length_range[1])
        ]
        
    # Sentiment Score Filter (-1.0 to +1.0)
    if 'sentiment_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['sentiment_score'] >= score_range[0]) & 
            (filtered_df['sentiment_score'] <= score_range[1])
        ]
        
    # Global Search Text Filter
    if search_query and 'text' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['text'].str.contains(search_query, case=False, na=False)]
        
    return filtered_df

# ==========================================
# 2. SEABORN PLOTS CONFIG (Purani charts.py ab yahan hai)
# ==========================================
sns.set_theme(style="darkgrid")
plt.rcParams.update({
    'figure.max_open_warning': 0, 
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'figure.facecolor': '#0e1117',
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'axes.facecolor': '#1e2430'
})

def plot_pie_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty and 'newsgroup' in df.columns:
        data = df['newsgroup'].value_counts().head(7)
        if not data.empty:
            ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, textprops={'color': 'white'}, colors=sns.color_palette("pastel", len(data)))
            ax.set_title("1. Proportional Mental Health Distribution (Pie)", pad=15, color='white')
    return fig

def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty and 'sentiment_score' in df.columns:
        sns.histplot(data=df, x='sentiment_score', hue='sentiment_category', kde=True, ax=ax, palette="magma", multiple="stack", bins=20)
        ax.set_title("2. Psychological Sentiment Density Spectrum (Histogram)", pad=15, color='white')
    return fig

def plot_bar_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty and 'newsgroup' in df.columns:
        sns.barplot(data=df, x='newsgroup', y='word_count', ax=ax, palette="flare", errorbar=None, estimator=np.mean)
        ax.set_title("3. Volume Density Mapping per Stress Group (Bar)", pad=15, color='white')
        plt.xticks(rotation=25, ha='right')
    return fig

def plot_area_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df
