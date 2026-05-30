import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Standard Premium Aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.max_open_warning': 0, 
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13
})

# 1. Pie Chart
def plot_pie_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = df['newsgroup'].value_counts().head(5)
    if not data.empty:
        ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(data)))
        ax.set_title("1. Proportional Volumetric Distribution (Top 5 Categories)", pad=15)
    return fig

# 2. Histogram
def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df['text_length'], kde=True, ax=ax, color="#1f77b4", bins=25)
    ax.set_title("2. Frequency Density Analysis of Text Lengths", pad=15)
    ax.set_xlabel("Text Frame Size (Characters)")
    ax.set_ylabel("Density Load")
    return fig

# 3. Line Chart
def plot_line_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sorted_df = df.sort_values(by='article_id').head(40)
    ax.plot(range(len(sorted_df)), sorted_df['word_count'].cumsum(), marker="o", markersize=4, linestyle="-", color="#ff7f0e", linewidth=2)
    ax.set_title("3. Cumulative Word Metrics Progression Over Sequence", pad=15)
    ax.set_xlabel("Sequential Sample Progression Index")
    ax.set_ylabel("Aggregated Words Volume")
    return fig

# 4. Bar Chart
def plot_bar_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=df, x='newsgroup', y='word_count', ax=ax, palette="plasma", errorbar=None, estimator=np.mean)
    ax.set_title("4. Comparative Assessment: Mean Word Count across Classes", pad=15)
    ax.set_xlabel("Newsgroup Classification")
    ax.set_ylabel("Mathematical Mean Length")
    plt.xticks(rotation=25, ha='right')
    return fig

# 5. Scatter Plot
def plot_scatter_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x='word_count', y='sentiment_score', ax=ax, alpha=0.6, color="#2ca02c", edgecolor="w")
    ax.set_title("5. Inter-variable Analysis: Metrics vs Sentiment Axis", pad=15)
    ax.set_xlabel("Absolute Word Metric Counts")
    ax.set_ylabel("Calculated Sentiment Space")
    return fig

# 6. Box Plot
def plot_box_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x='newsgroup', y='sentiment_score', ax=ax, palette="Set2")
    ax.set_title("6. Statistical Variability Mapping & Outlier Detection", pad=15)
    ax.set_xlabel("Target Newsgroup Domain")
    ax.set_ylabel("Sentiment Standard Spread")
    plt.xticks(rotation=25, ha='right')
    return fig

# 7. Heatmap
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_cols = ['word_count', 'text_length', 'avg_word_length', 'sentiment_score']
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if valid_cols:
        corr = df[valid_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True, square=True)
        ax.set
