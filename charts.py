import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Standard Premium Aesthetics for Healthcare/Psychological Dashboards
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

# 1. Pie Chart - Proportional Mental Health Distribution
def plot_pie_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if 'newsgroup' in df.columns and not df.empty:
        data = df['newsgroup'].value_counts().head(7)
        if not data.empty:
            ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, 
                   textprops={'color': 'white'}, colors=sns.color_palette("pastel", len(data)))
            ax.set_title("1. Proportional Mental Health Distribution (Pie)", pad=15, color='white')
    return fig

# 2. Histogram - Psychological Sentiment Density Spectrum
def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if 'sentiment_score' in df.columns and not df.empty:
        sns.histplot(data=df, x='sentiment_score', hue='sentiment_category', kde=True, ax=ax, palette="magma", multiple="stack", bins=20)
        ax.set_title("2. Psychological Sentiment Density Spectrum (Histogram)", pad=15, color='white')
        ax.set_xlabel("VADER Sentiment Score Axis", color='white')
        ax.set_ylabel("Count / Density", color='white')
    return fig

# 3. Bar Chart - Volume Density Mapping per Stress Group
def plot_bar_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if 'newsgroup' in df.columns and not df.empty:
        sns.barplot(data=df, x='newsgroup', y='word_count', ax=ax, palette="flare", errorbar=None, estimator=np.mean)
        ax.set_title("3. Volume Density Mapping per Stress Group (Bar)", pad=15, color='white')
        ax.set_xlabel("Assigned Psychological Topics", color='white')
        ax.set_ylabel("Mathematical Mean Word Count", color='white')
        plt.xticks(rotation=25, ha='right')
    return fig

# 4. Area Chart - Discourse Temporal Post Velocity Wave
def plot_area_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        sorted_subset = df.sort_values(by='article_id').head(40)
        x_axis = range(len(sorted_subset))
        ax.fill_between(x_axis, sorted_subset['text_length'], color="#00bcd4", alpha=0.3)
        ax.plot(x_axis, sorted_subset['text_length'], color="#00bcd4", alpha=1.0, linewidth=2)
        ax.set_title("4. Discourse Temporal Post Velocity Wave (Area)", pad=15, color='white')
        ax.set_xlabel("Sequential Post Timeline Index", color='white')
        ax.set_ylabel("Volume Count / Character Frame Size", color='white')
    return fig

# 5. Box Plot - Emotional Dispersion Variance Bounds (Yeh crash point tha, ab full safe hai)
def plot_box_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if 'newsgroup' in df.columns and 'sentiment_score' in df.columns and not df.empty:
            sns.boxplot(data=df, y='newsgroup', x='sentiment_score', ax=ax, palette="Set2", hue='newsgroup', legend=False)
            ax.set_title("5. Emotional Dispersion Variance Bounds (Box Plot)", pad=15, color='white')
            ax.set_xlabel("VADER Value Metrics Across Strategic Domains", color='white')
            ax.set_ylabel("Assigned Topic Classifications", color='white')
    except Exception as e:
        ax.text(0.5, 0.5, f"Waiting for filter adjustment... \n{str(e)}", color='orange', ha='center', va='center')
    return fig

# 6. Scatter Plot - Bivariate Demographic Scale Layout
def plot_scatter_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if 'word_count' in df.columns and 'sentiment_score' in df.columns and not df.empty:
            sns.scatterplot(data=df, x='word_count', y='sentiment_score', hue='newsgroup', ax=ax, alpha=0.7, palette="viridis", edgecolor="w")
            ax.set_title("6. Bivariate Demographic Scale Layout (Scatter)", pad=15, color='white')
            ax.set_xlabel("Anxiety Word Pattern Counts", color='white')
            ax.set_ylabel("Calculated Sentiment Space Scale", color='white')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    except Exception as e:
        pass
    return fig

# 7. Heatmap - Multi-Parametric Feature Inter-Correlation Matrix
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_cols = ['word_count', 'text_length', 'avg_word_length', 'sentiment_score']
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if valid_cols and len(df) > 1:
        corr = df[valid_cols].corr().fillna(0)
        sns.heatmap(corr, annot=True, cmap="icefire", fmt=".2f", ax=ax, cbar=True, square=True)
        ax.set_title("7. Multi-Parametric Feature Inter-Correlation Matrix (Heatmap)", pad=15, color='white')
    return fig

# 8. Line Chart - Cumulative Expression Progression Track
def plot_line_chart(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not df.empty:
        sorted_df = df.sort_values(by='article_id').head(35)
        ax.plot(range(len(sorted_df)), sorted_df['word_count'].cumsum(), marker="o", markersize=3, linestyle="-", color="#ff5722", linewidth=2)
        ax.set_title("8. Cumulative Expression Progression Track (Line)", pad=15, color='white')
        ax.set_xlabel("Sequential Sample Tracking Order", color='white')
        ax.set_ylabel("Aggregated Words Volume Metric", color='white')
    return fig

# 9. Count Plot - Domain Post Density Distribution Map
def plot_count_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if 'newsgroup' in df.columns and not df.empty:
        sns.countplot(data=df, y='newsgroup', ax=ax, palette="crest", order=df['newsgroup'].value_counts().index, hue='newsgroup', legend=False)
        ax.set_title("9. Domain Post Density Distribution Map (Count Plot)", pad=15, color='white')
        ax.set_xlabel("Absolute Volumetric Load", color='white')
        ax.set_ylabel("Topic Classification Classes", color='white')
    return fig

# 10. Violin Plot - Probability Density Profile Interface
def plot_violin_plot(df):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if 'newsgroup' in df.columns and 'text_length' in df.columns and not df.empty:
            sns.violinplot(data=df, x='text_length', y='newsgroup', ax=ax, palette="pastel", inner="quartile", hue='newsgroup', legend=False)
            ax.set_title("10. Probability Density Profile Interface (Violin Plot)", pad=15, color='white')
            ax.set_xlabel("Text Payload Character Scale Length", color='white')
            ax.set_ylabel("Psychological Category", color='white')
    except Exception as e:
        pass
    return fig
