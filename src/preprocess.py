import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' url ', text)
    text = re.sub(r'\$[\d,]+', ' moneysym ', text)
    text = re.sub(r'\d+', ' num ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(words)

def extract_features(text):
    return {
        'has_url':       int(bool(re.search(r'http|www', text.lower()))),
        'has_money':     int(bool(re.search(r'\$|\bfree\b|\bwin\b|\bcash\b', text.lower()))),
        'has_caps':      int(sum(1 for c in text if c.isupper()) > 10),
        'exclaim_count': text.count('!'),
        'word_count':    len(text.split()),
        'char_count':    len(text),
    }
