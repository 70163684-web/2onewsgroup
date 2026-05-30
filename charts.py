import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Establishing standard premium color systems
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'figure.max_open_warning': 0, 
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12
})

# 1. Pie Chart
def plot_pie_chart(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df['newsgroup'].value_counts().head(5)
    if not data.empty:
        ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(data)))
        ax.set_title("Proportional Volumetric Distribution (Top 5 Categories)")
    return fig

# 2. Histogram
def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df['text_length'], kde=True, ax=ax, color="#1f77b4", bins=25)
    ax.set_title("Frequency Density Analysis of Text Lengths")
    ax.set_xlabel("Text Frame Size (Characters)")
    ax.set_ylabel("Density Load")
    return fig

# 3. Line Chart
def plot_line_chart(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_df = df.sort_values(by='article_id').head(40)
    ax.plot(range(len(sorted_df)), sorted_df['word_count'].cumsum(), marker="", linestyle="-", color="#ff7f0e", linewidth=2)
    ax.set_title("Cumulative Word Metrics Progression Over Sequence")
    ax.set_xlabel("Sequential Sample Progression Index")
    ax.set_ylabel("Aggregated Words Volume")
    return fig

# 4. Bar Chart
def plot_bar_chart(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='newsgroup', y='word_count', ax=ax, palette="plasma", errorbar=None, estimator=np.mean)
    ax.set_title("Comparative Assessment: Mean Word Count across Categories")
    ax.set_xlabel("Newsgroup Classification")
    ax.set_ylabel("Mathematical Mean Length")
    plt.xticks(rotation=35, ha='right')
    return fig

# 5. Scatter Plot
def plot_scatter_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='word_count', y='sentiment_score', ax=ax, alpha=0.6, color="#2ca02c", edgecolor="w")
    ax.set_title("Inter-variable Analysis: Metrics vs Sentiment Axis")
    ax.set_xlabel("Absolute Word Metric Counts")
    ax.set_ylabel("Calculated Sentiment Space")
    return fig

# 6. Box Plot
def plot_box_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='newsgroup', y='sentiment_score', ax=ax, palette="Set2")
    ax.set_title("Statistical Variability Mapping & Outlier Detection")
    ax.set_xlabel("Target Newsgroup Domain")
    ax.set_ylabel("Sentiment Standard Spread")
    plt.xticks(rotation=35, ha='right')
    return fig

# 7. Heatmap
def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    numeric_cols = ['article_id', 'word_count', 'text_length', 'avg_word_length', 'sentiment_score']
    valid_cols = [c for c in numeric_cols if c in df.columns]
    if valid_cols:
        corr = df[valid_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=True, square=True)
        ax.set_title("Feature Inter-correlation Matrix Heatmap")
    return fig

# 8. Area Chart
def plot_area_chart(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sorted_subset = df.sort_values(by='article_id').head(35)
    x_axis = range(len(sorted_subset))
    ax.fill_between(x_axis, sorted_subset['avg_word_length'], color="#9467bd", alpha=0.4)
    ax.plot(x_axis, sorted_subset['avg_word_length'], color="#9467bd", alpha=1.0, linewidth=1.5)
    ax.set_title("Structural Progression Profile: Average Word Lengths")
    ax.set_xlabel("Sample Timeline Index")
    ax.set_ylabel("Average Frame Scale")
    return fig

# 9. Count Plot
def plot_count_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='newsgroup', ax=ax, palette="magma", order=df['newsgroup'].value_counts().index)
    ax.set_title("Absolute Density Volumes per Newsgroup Instance")
    ax.set_xlabel("Target Classification Group")
    ax.set_ylabel("Total Post Count Volumetrics")
    plt.xticks(rotation=35, ha='right')
    return fig

# 10. Violin Plot
def plot_violin_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x='newsgroup', y='text_length', ax=ax, palette="muted", inner="quartile")
    ax.set_title("Probability Density Distribution Matrix across Classes")
    ax.set_xlabel("Newsgroup Domain")
    ax.set_ylabel("Text Payload Length Scales")
    plt.xticks(rotation=35, ha='right')
    return fig