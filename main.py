import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import string
import warnings
warnings.filterwarnings("ignore")

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

df = pd.read_csv("spam_ham.csv", encoding='latin1', on_bad_lines='skip', names=["label", "text"])

print("5 mensajes aleatorios:")
print(df.sample(5))

print(f"\nTotal de mensajes: {len(df)}")

sns.countplot(data=df, x='label')
plt.title("Distribución de mensajes spam vs ham")
plt.show()

df['length'] = df['text'].apply(len)

sns.kdeplot(data=df[df['label'] == 'spam'], x='length', fill=True)
plt.title("Densidad de longitud de mensajes SPAM")
plt.show()

sns.kdeplot(data=df[df['label'] == 'ham'], x='length', fill=True)
plt.title("Densidad de longitud de mensajes HAM")
plt.show()

def get_top_words(texts, n=20):
    all_words = ' '.join(texts).lower().split()
    return Counter(all_words).most_common(n)

spam_words = get_top_words(df[df['label'] == 'spam']['text'])
print("\nTop 20 palabras SPAM:", spam_words)

ham_words = get_top_words(df[df['label'] == 'ham']['text'])
print("\nTop 20 palabras HAM:", ham_words)

spam_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'spam']['text']))
ham_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'ham']['text']))

plt.figure(figsize=(10,5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.title("WordCloud - SPAM")
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(ham_wc, interpolation='bilinear')
plt.title("WordCloud - HAM")
plt.axis("off")
plt.show()

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

df['processed'] = df['text'].apply(preprocess)
df['processed_text'] = df['processed'].apply(lambda tokens: ' '.join(tokens))

print(f"\nTotal de mensajes tras preprocesamiento: {len(df)}")

sns.countplot(data=df, x='label')
plt.title("Distribución de mensajes tras preprocesamiento")
plt.show()

df['processed_length'] = df['processed_text'].apply(len)

sns.kdeplot(data=df[df['label'] == 'spam'], x='processed_length', fill=True)
plt.title("Densidad de longitud (procesado) - SPAM")
plt.show()

sns.kdeplot(data=df[df['label'] == 'ham'], x='processed_length', fill=True)
plt.title("Densidad de longitud (procesado) - HAM")
plt.show()

spam_words_proc = get_top_words(df[df['label'] == 'spam']['processed_text'])
ham_words_proc = get_top_words(df[df['label'] == 'ham']['processed_text'])
print("\nTop 20 palabras SPAM (procesado):", spam_words_proc)
print("\nTop 20 palabras HAM (procesado):", ham_words_proc)

spam_wc_proc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'spam']['processed_text']))
ham_wc_proc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'ham']['processed_text']))

plt.figure(figsize=(10,5))
plt.imshow(spam_wc_proc, interpolation='bilinear')
plt.title("WordCloud - SPAM (procesado)")
plt.axis("off")
plt.show()

plt.figure(figsize=(10,5))
plt.imshow(ham_wc_proc, interpolation='bilinear')
plt.title("WordCloud - HAM (procesado)")
plt.axis("off")
plt.show()
