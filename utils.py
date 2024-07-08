import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to clean text data
def wordopt(text):
    text = text.lower()
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


