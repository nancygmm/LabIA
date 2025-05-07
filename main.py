import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

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
