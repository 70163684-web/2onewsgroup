import tarfile
import re
import pandas as pd
import numpy as np

def load_and_process_corpus(tar_path="20news-bydate.tar.gz"):
    """
    Robustly extracts architectural properties from the archive file 
    handling any compression variations cleanly to prevent server crashes.
    """
    parsed_data = []
    
    # Attempt parsing with multiple decompression modes fallback
    modes = ["r:gz", "r:", "r|gz"]
    tar = None
    for mode in modes:
        try:
            tar = tarfile.open(tar_path, mode)
            break
        except Exception:
            continue
            
    if tar is None:
        # Check fallback local name structure
        try:
            tar = tarfile.open("20news-bydate.tar", "r:")
        except Exception:
            return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength", "SentimentScore"])

    try:
        for member in tar.getmembers():
            if member.isfile() and ("train" in member.name.lower() or "test" in member.name.lower()):
                path_segments = member.name.split('/')
                if len(path_segments) >= 3:
                    split_type = "Train Split" if "train" in path_segments[0].lower() else "Test Split"
                    news_category = path_segments[1]
                    
                    f = tar.extractfile(member)
                    if f is not None:
                        raw_content = f.read().decode('utf-8', errors='ignore')
                        
                        subject_search = re.search(r'^Subject:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        lines_search = re.search(r'^Lines:\s*(\d+)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        org_search = re.search(r'^Organization:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        
                        email_subject = subject_search.group(1).strip() if (subject_search and subject_search.group(1)) else "Untitled Document"
                        total_lines = int(lines_search.group(1)) if (lines_search and lines_search.group(1)) else len(raw_content.splitlines())
                        organization = org_search.group(1).strip() if (org_search and org_search.group(1)) else "Unknown Workspace"
                        
                        header_boundary = raw_content.find('\n\n')
                        clean_body = raw_content[header_boundary:].strip() if header_boundary != -1 else raw_content
                        
                        parsed_data.append({
                            "BaseSplit": split_type,
                            "Category": news_category,
                            "Subject": email_subject,
                            "Lines": total_lines,
                            "Organization": organization,
                            "RawText": clean_body
                        })
        tar.close()
    except Exception:
        if tar:
            tar.close()

    if not parsed_data:
        return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength", "SentimentScore"])
        
    df = pd.DataFrame(parsed_data)
    
    # Text Normalization Engineering Pipeline
    df['CleanText'] = df['RawText'].str.lower()
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) 
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\d+', '', str(x)))      
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    df['WordCount'] = df['CleanText'].apply(lambda x: len(x.split()))
    df['CharCount'] = df['CleanText'].apply(lambda x: len(str(x)))
    df['AvgWordLength'] = df.apply(lambda row: row['CharCount'] / row['WordCount'] if row['WordCount'] > 0 else 0, axis=1)
    
    # Explicit 4-Quadrant Partition Strategy
    def segment_partitions(row):
        if row['BaseSplit'] == "Train Split":
            return "Train Split (Standard Vol)" if row['WordCount'] < 250 else "Train Split (Dense Matrix)"
        else:
            return "Test Split (Standard Vol)" if row['WordCount'] < 250 else "Test Split (Dense Matrix)"
            
    df['Split'] = df.apply(segment_partitions, axis=1)
    
    # Fast Lexicon VADER-inspired Rule Set
    pos_words = {'good', 'great', 'excellent', 'agree', 'right', 'support', 'true', 'thanks', 'benefit', 'solve', 'perfect'}
    neg_words = {'bad', 'wrong', 'error', 'fail', 'problem', 'severe', 'claim', 'disagree', 'attack', 'kill', 'hate'}
    
    def score_sentiment(text):
        tokens = text.split()
        p = sum(1 for t in tokens if t in pos_words)
        n = sum(1 for t in tokens if t in neg_words)
        return "Positive" if p > n else ("Negative" if n > p else "Neutral")
        
    df['SentimentScore'] = df['CleanText'].apply(score_sentiment)
    return df[df['WordCount'] >= 1].reset_index(drop=True)

def extract_advanced_vocabulary(df):
    if df.empty or 'CleanText' not in df.columns:
        return pd.DataFrame(columns=['Word', 'Frequency'])
        
    base_stopwords = {
        'the', 'and', 'of', 'to', 'is', 'in', 'for', 'on', 'with', 'a', 'an', 'this', 'are', 
        'or', 'at', 'from', 'it', 'that', 'by', 'be', 'as', 'was', 'have', 'not', 'but', 'you', 'i', 'he', 'they'
    }
    
    token_frequencies = {}
    for passage in df['CleanText'].head(1500):  
        tokens = str(passage).split()
        for token in tokens:
            if token not in base_stopwords and len(token) > 4:
                token_frequencies[token] = token_frequencies.get(token, 0) + 1
                
    return pd.DataFrame(sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True), columns=['Word', 'Frequency']).head(15)
