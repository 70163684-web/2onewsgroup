import tarfile
import re
import pandas as pd

def load_and_process_corpus(tar_path="20news-bydate.tar"):
    """
    Directly streams and extracts the archive in memory.
    Automatically handles both compressed (.tar.gz) and uncompressed (.tar) formats.
    """
    parsed_data = []
    
    # Pehle try karein standard tar mode ('r:'), agar fail ho toh compressed mode ('r:gz') use karein
    try:
        tar = tarfile.open(tar_path, "r:")
    except Exception:
        try:
            tar = tarfile.open(tar_path, "r:gz")
        except Exception as e:
            # Agar file name badla hua ho (e.g. tar.gz), toh check karein
            if "tar.gz" in tar_path or "gz" in str(e):
                tar = tarfile.open("20news-bydate.tar.gz", "r:gz")
            else:
                return pd.DataFrame(columns=["Split", "Category", "Subject", "RawText", "CleanText", "WordCount", "CharCount"])

    try:
        for member in tar.getmembers():
            # Read from both train and test subsets automatically
            if member.isfile() and ("20news-bydate-train" in member.name or "20news-bydate-test" in member.name):
                path_segments = member.name.split('/')
                if len(path_segments) >= 3:
                    split_type = "Train Split" if "train" in path_segments[0] else "Test Split"
                    news_category = path_segments[1]
                    
                    f = tar.extractfile(member)
                    if f is not None:
                        raw_content = f.read().decode('utf-8', errors='ignore')
                        
                        # Structural feature separation using Regex
                        subject_search = re.search(r'^Subject:\s*(.*)$', raw_content, re.MULTILINE | re.IGNORECASE)
                        email_subject = subject_search.group(1).strip() if subject_search else "Untitled Document"
                        
                        # Clean institutional headers boundary split
                        header_boundary = raw_content.find('\n\n')
                        clean_body = raw_content[header_boundary:].strip() if header_boundary != -1 else raw_content
                        
                        parsed_data.append({
                            "Split": split_type,
                            "Category": news_category,
                            "Subject": email_subject,
                            "RawText": clean_body
                        })
        tar.close()
    except Exception:
        pass

    # DataFrame construction
    if not parsed_data:
        return pd.DataFrame(columns=["Split", "Category", "Subject", "RawText", "CleanText", "WordCount", "CharCount"])
        
    df = pd.DataFrame(parsed_data)
    
    # Text Token Analysis Pipeline
    df['CleanText'] = df['RawText'].str.lower()
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x))) 
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\d+', '', str(x)))      
    df['CleanText'] = df['CleanText'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    
    # Generate Numerical Features
    df['WordCount'] = df['CleanText'].apply(lambda x: len(x.split()))
    df['CharCount'] = df['CleanText'].apply(lambda x: len(x))
    df['AvgWordLength'] = df.apply(lambda row: row['CharCount'] / row['WordCount'] if row['WordCount'] > 0 else 0, axis=1)
    
    return df[df['WordCount'] > 5].reset_index(drop=True)

def extract_advanced_vocabulary(df):
    """Calculates frequency distributions dynamically for charts"""
    base_stopwords = {
        'the', 'and', 'of', 'to', 'is', 'in', 'for', 'on', 'with', 'a', 'an', 'this', 'are', 
        'or', 'at', 'from', 'it', 'that', 'by', 'be', 'as', 'was', 'have', 'not', 'but', 'you', 'i', 'he', 'they'
    }
    
    token_frequencies = {}
    for passage in df['CleanText']:
        tokens = passage.split()
        for token in tokens:
            if token not in base_stopwords and len(token) > 4:
                token_frequencies[token] = token_frequencies.get(token, 0) + 1
                
    sorted_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
    return pd.DataFrame(sorted_tokens, columns=['Word', 'Frequency']).head(15)
