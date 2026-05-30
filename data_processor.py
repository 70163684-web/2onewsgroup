import tarfile
import re
import pandas as pd

def load_and_process_corpus(tar_path="20news-bydate.tar"):
    """
    Highly optimized memory parser that loads the COMPLETE 20 Newsgroups dataset.
    Extracts core text statistics, email components, and partitions.
    """
    parsed_data = []
    
    # Automatic dynamic file detection extension fallback
    try:
        tar = tarfile.open(tar_path, "r:")
    except Exception:
        try:
            tar = tarfile.open(tar_path, "r:gz")
        except Exception:
            try:
                tar = tarfile.open("20news-bydate.tar.gz", "r:gz")
            except Exception:
                return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength"])

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
                        
                        # Extract structural features via high-speed RegEx
                        subject_search = re.search(r'^Subject:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        lines_search = re.search(r'^Lines:\s*(\d+)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        org_search = re.search(r'^Organization:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        
                        email_subject = subject_search.group(1).strip() if subject_search else "Untitled Document"
                        total_lines = int(lines_search.group(1)) if lines_search else 0
                        organization = org_search.group(1).strip() if org_search else "Unknown Organization"
                        
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
        return pd.DataFrame(columns=["Split", "Category", "Subject", "Lines", "Organization", "RawText", "CleanText", "WordCount", "CharCount", "AvgWordLength"])
        
    df = pd.DataFrame(parsed_data)
    
    # Fast NLP Preprocessing Strings Pipeline
    df['CleanText'] = df['RawText'].str.lower()
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) 
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\d+', '', str(x)))      
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    # Calculate Statistical Text Metrics
    df['WordCount'] = df['CleanText'].apply(lambda x: len(x.split()))
    df['CharCount'] = df['CleanText'].apply(lambda x: len(x))
    df['AvgWordLength'] = df.apply(lambda row: row['CharCount'] / row['WordCount'] if row['WordCount'] > 0 else 0, axis=1)
    
    # Keep true files to ensure dynamic calculations match
    return df[df['WordCount'] >= 1].reset_index(drop=True)

def extract_advanced_vocabulary(df):
    """Dynamically parses the vocabulary counts for chart widgets securely."""
    if df.empty or 'CleanText' not in df.columns:
        return pd.DataFrame(columns=['Word', 'Frequency'])
        
    base_stopwords = {
        'the', 'and', 'of', 'to', 'is', 'in', 'for', 'on', 'with', 'a', 'an', 'this', 'are', 
        'or', 'at', 'from', 'it', 'that', 'by', 'be', 'as', 'was', 'have', 'not', 'but', 'you', 'i', 'he', 'they'
    }
    
    token_frequencies = {}
    for passage in df['CleanText'].head(1200):  
        tokens = str(passage).split()
        for token in tokens:
            if token not in base_stopwords and len(token) > 4:
                token_frequencies[token] = token_frequencies.get(token, 0) + 1
                
    sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_tokens, columns=['Word', 'Frequency']).head(15)
