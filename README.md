# 📰 20 Newsgroups Text Mining & EDA Dashboard

An advanced, professional-grade Exploratory Data Analysis (EDA) and Text Analytics Dashboard built using Python, Streamlit, Matplotlib, and Seaborn. This system provides interactive multi-dimensional data filtering and visual analytics over the standard **20 Newsgroups dataset**.

---

## 🚀 Features & Functionalities

### 1. Multi-Dimensional Interactive Filters (Sidebar)
* **Category Dropdown:** Multi-select element to dynamically isolate or combine various newsgroup classes (e.g., `sci.med`, `comp.graphics`).
* **Text Length Range Slider:** Continuous interval selection targeting document character payload spans.
* **Sentiment Index Slider:** Continuous float filter adjusting text sentiment ranges dynamically from `-1.0` to `+1.0`.
* **Global Search Text Input:** Live case-insensitive keyword/token substring pattern matcher.
* **Global Reset System:** A dynamic clearing engine that resets all visualization parameters to base values instantly.

### 2. 10 Mandatory Charts (Synchronized Grid Layout)
The interface employs a balanced 2-column layout displaying the following analytical plots simultaneously:
1. **Proportional Pie Matrix:** Volumetric class percentage distributions.
2. **Payload Distribution Histogram:** Density mapping of string content sizes.
3. **Trend Accumulation Line Graphic:** Cumulative text volume progression.
4. **Categorical Length Bar Graph:** Cross-class mean word comparisons.
5. **Scatter Relational Mapping:** Structural inter-variable token layouts.
6. **Outlier Dispersion Box Plot:** Whisker distribution of calculated scores.
7. **Correlation Heatmap Matrix:** Dynamic linear interaction weights of numerical parameters.
8. **Progression Area Representation:** Volumetric baseline trends over sample counts.
9. **Absolute Frequency Count Profile:** Order-optimized ranking of sub-categories.
10. **Density Profile Violin Interface:** Probability densities mapping across textual domains.

---

## 📂 Project Structure & Architecture

```text
/20news_dashboard_project/
│
├── data/                               
│   └── 20news-bydate.csv               <-- Enriched text features output (No Renaming)
│
├── 20news-bydate.tar.gz                <-- Original unchanged compressed dataset
├── make_csv.py                         <-- Custom automated processing ETL pipeline
├── filters.py                          <-- Multi-variable synchronous filter module
├── charts.py                           <-- Visual framework containing all 10 plot models
├── app.py                              <-- Main UI runtime execution application engine
├── requirements.txt                    <-- Complete third-party dependency list
└── README.md                           <-- Comprehensive documentation (This File)