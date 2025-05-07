import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("spam_ham.csv", encoding='latin1', sep=';', names=["label", "text"], skiprows=1)
df = df.dropna()
df = df[df['label'].isin(['ham', 'spam'])]

print("\nCANTIDAD TOTAL DE MENSAJES:")
print(len(df))

print("\nDISTRIBUCIÓN Y PROPORCIÓN DE MENSAJES SPAM VS HAM:")
print(df['label'].value_counts(normalize=True))

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label')
plt.title("Distribución de mensajes spam vs ham")
plt.show()

df['length'] = df['text'].apply(len)

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df[df['label'] == 'spam'], x='length', fill=True)
plt.title("Densidad de longitud de mensajes SPAM")
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df[df['label'] == 'ham'], x='length', fill=True)
plt.title("Densidad de longitud de mensajes HAM")
plt.show()

def get_top_words(texts, n=20):
    all_words = ' '.join(texts).lower().split()
    return Counter(all_words).most_common(n)

print("\nTOP 20 PALABRAS MÁS FRECUENTES EN MENSAJES SPAM:")
spam_words = get_top_words(df[df['label'] == 'spam']['text'])
for word, count in spam_words:
    print(f"{word}: {count}")

print("\nTOP 20 PALABRAS MÁS FRECUENTES EN MENSAJES HAM:")
ham_words = get_top_words(df[df['label'] == 'ham']['text'])
for word, count in ham_words:
    print(f"{word}: {count}")

spam_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'spam']['text']))
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - SPAM")
plt.show()

ham_wc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'ham']['text']))
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - HAM")
plt.show()

print("\n--- PREPROCESAMIENTO DE TEXTO CON NLTK ---")
print("Transformaciones aplicadas:")
print("1. Conversión a minúsculas: para unificar el texto y evitar duplicados como 'Free' y 'free'.")
print("2. Eliminación de signos de puntuación: para enfocarnos en las palabras importantes.")
print("3. Tokenización: para dividir el texto en palabras individuales.")
print("4. Eliminación de stopwords: para eliminar palabras comunes sin valor informativo como 'the', 'and', etc.")
print("5. Stemming con PorterStemmer: para reducir palabras a su raíz (por ejemplo, 'running' -> 'run').")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return tokens

df['processed'] = df['text'].apply(preprocess)
df['processed_text'] = df['processed'].apply(lambda tokens: ' '.join(tokens))

print("\nCANTIDAD TOTAL DE MENSAJES (POST-PROCESAMIENTO):")
print(len(df))

print("\nPROPORCIÓN DE SPAM VS HAM (POST-PROCESAMIENTO):")
print(df['label'].value_counts(normalize=True))

df['processed_length'] = df['processed_text'].apply(len)

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df[df['label'] == 'spam'], x='processed_length', fill=True)
plt.title("Densidad de longitud (procesado) - SPAM")
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(data=df[df['label'] == 'ham'], x='processed_length', fill=True)
plt.title("Densidad de longitud (procesado) - HAM")
plt.show()

print("\nTOP 20 PALABRAS MÁS FRECUENTES EN SPAM (POST-PROCESAMIENTO):")
spam_words_proc = get_top_words(df[df['label'] == 'spam']['processed_text'])
for word, count in spam_words_proc:
    print(f"{word}: {count}")

print("\nTOP 20 PALABRAS MÁS FRECUENTES EN HAM (POST-PROCESAMIENTO):")
ham_words_proc = get_top_words(df[df['label'] == 'ham']['processed_text'])
for word, count in ham_words_proc:
    print(f"{word}: {count}")

spam_wc_proc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'spam']['processed_text']))
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc_proc, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - SPAM (procesado)")
plt.show()

ham_wc_proc = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df[df['label'] == 'ham']['processed_text']))
plt.figure(figsize=(10, 5))
plt.imshow(ham_wc_proc, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - HAM (procesado)")
plt.show()

print("\nEjemplo de mensaje original y preprocesado:")
print(df[['text', 'processed_text']].sample(3))
