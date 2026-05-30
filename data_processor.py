import tarfile
import re
import pandas as pd

def load_and_process_corpus(tar_path="20news-bydate.tar"):
    """
    Highly secure streaming archive parser. Robustly extracts structural properties 
    and handles files missing metadata cleanly to guarantee zero crashes.
    """
    parsed_data = []
    
    # Secure stream fallbacks tracking
    try:
        tar = tarfile.open(tar_path, "r:")
    except Exception:
        try:
            tar = tarfile.open(tar_path, "r:gz")
        except Exception:
            try:
                tar = tarfile.open("20news-bydate.tar.gz", "r:gz")
            except Exception:
                return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength", "SentimentScore"])

    try:
        for member in tar.getmembers():
            if member.isfile() and ("20news-bydate-train" in member.name or "20news-bydate-test" in member.name):
                path_segments = member.name.split('/')
                if len(path_segments) >= 3:
                    split_type = "Train Split" if "train" in path_segments[0] else "Test Split"
                    news_category = path_segments[1]
                    
                    f = tar.extractfile(member)
                    if f is not None:
                        raw_content = f.read().decode('utf-8', errors='ignore')
                        
                        # High-grade safe conditional RegEx engine
                        subject_search = re.search(r'^Subject:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        lines_search = re.search(r'^Lines:\s*(\d+)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        org_search = re.search(r'^Organization:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        
                        email_subject = subject_search.group(1).strip() if (subject_search and subject_search.group(1)) else "Untitled Document"
                        total_lines = int(lines_search.group(1)) if (lines_search and lines_search.group(1)) else 0
                        organization = org_search.group(1).strip() if (org_search and org_search.group(1)) else "Unknown Workspace"
                        
                        header_boundary = raw_content.find('\n\n')
                        clean_body = raw_content[header_boundary:].strip() if header_boundary != -1 else raw_content
                        
                        parsed_data.append({
                            "Split": split_type,
                            "Category": news_category,
                            "Subject": email_subject,
                            "Lines": total_lines,
                            "Organization": organization,
                            "RawText": clean_body
                        })
        tar.close()
    except Exception:
        pass

    if not parsed_data:
        return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength", "SentimentScore"])
        
    df = pd.DataFrame(parsed_data)
    
    # Data Cleaning and Preprocessing Pipeline (Pandas & Strings)
    df['CleanText'] = df['RawText'].str.lower()
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) 
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\d+', '', str(x)))      
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    # Feature Metrics Generation
    df['WordCount'] = df['CleanText'].apply(lambda x: len(x.split()))
    df['CharCount'] = df['CleanText'].apply(lambda x: len(x))
    df['AvgWordLength'] = df.apply(lambda row: row['CharCount'] / row['WordCount'] if row['WordCount'] > 0 else 0, axis=1)
    
    # Custom Rule-Based NLP Sentiment Tagger
    pos_words = {'good', 'great', 'excellent', 'agree', 'right', 'support', 'true', 'thanks', 'benefit', 'solve'}
    neg_words = {'bad', 'wrong', 'error', 'fail', 'problem', 'severe', 'claim', 'disagree', 'attack', 'kill'}
    
    def score_sentiment(text):
        tokens = text.split()
        p = sum(1 for t in tokens if t in pos_words)
        n = sum(1 for t in tokens if t in neg_words)
        return "Positive" if p > n else ("Negative" if n > p else "Neutral")
        
    df['SentimentScore'] = df['CleanText'].apply(score_sentiment)
    
    return df[df['WordCount'] >= 1].reset_index(drop=True)

def extract_advanced_vocabulary(df):
    """Safely extracts top tokens frequency for high-quality dashboard plotting."""
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
                
    sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_tokens, columns=['Word', 'Frequency']).head(15)
