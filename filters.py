import pandas as pd

def apply_filters(df, selected_categories, length_range, score_range, search_query):
    """
    Applies multi-dimensional textual and continuous numerical filters simultaneously.
    """
    filtered_df = df.copy()
    
    # 1. Category multi-select filter
    if selected_categories and 'newsgroup' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['newsgroup'].isin(selected_categories)]
        
    # 2. Character text length interval slider
    if 'text_length' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['text_length'] >= length_range[0]) & 
            (filtered_df['text_length'] <= length_range[1])
        ]
        
    # 3. Sentiment numerical evaluation index slider
    if 'sentiment_score' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['sentiment_score'] >= score_range[0]) & 
            (filtered_df['sentiment_score'] <= score_range[1])
        ]
        
    # 4. Global string case-insensitive advanced keyword search
    if search_query and 'text' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['text'].str.contains(search_query, case=False, na=False)]
        
    return filtered_df