
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv("spam_ham.csv", encoding='latin1', sep=';', names=["label", "text"], skiprows=1)
df = df.dropna()
df = df[df['label'].isin(['ham', 'spam'])]

print("\nCANTIDAD TOTAL DE MENSAJES:", len(df))
print("\nDISTRIBUCIÓN DE CLASES (SPAM vs HAM):")
print(df['label'].value_counts(normalize=True))

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

X_train, X_test, y_train, y_test = train_test_split(
    df['processed'], df['label'], test_size=0.2, random_state=42
)

def contar_palabras(lista_tokens, etiquetas):
    spam_words, ham_words = [], []
    for tokens, label in zip(lista_tokens, etiquetas):
        if label == 'spam':
            spam_words.extend(tokens)
        else:
            ham_words.extend(tokens)
    return Counter(spam_words), Counter(ham_words)

spam_freq, ham_freq = contar_palabras(X_train, y_train)
P_spam = (y_train == 'spam').mean()
P_ham = 1 - P_spam
vocab = set(spam_freq) | set(ham_freq)
total_spam_words = sum(spam_freq.values())
total_ham_words = sum(ham_freq.values())

def P_w_given_class(word, class_freq, total_words):
    return (class_freq.get(word, 0) + 1) / (total_words + len(vocab))

def P_spam_given_words(words):
    probs = []
    for w in words:
        PwS = P_w_given_class(w, spam_freq, total_spam_words)
        PwH = P_w_given_class(w, ham_freq, total_ham_words)
        numerator = PwS * P_spam
        denominator = numerator + PwH * P_ham
        if denominator > 0:
            probs.append(numerator / denominator)
    if not probs:
        return 0.5
    product = np.prod(probs)
    inv_product = np.prod([1 - p for p in probs])
    return product / (product + inv_product)

def evaluar_modelo(threshold):
    y_pred = []
    for tokens in X_test:
        prob = P_spam_given_words(tokens)
        y_pred.append("spam" if prob >= threshold else "ham")
    cm = confusion_matrix(y_test, y_pred, labels=["spam", "ham"])
    precision = precision_score(y_test, y_pred, pos_label='spam')
    recall = recall_score(y_test, y_pred, pos_label='spam')
    f1 = f1_score(y_test, y_pred, pos_label='spam')
    print(f"\nEvaluación del modelo con threshold = {threshold}")
    print("--------------------------------------------------")
    print("Matriz de Confusión:")
    print(f"  Verdaderos Positivos (SPAM correctamente detectado): {cm[0,0]}")
    print(f"  Falsos Positivos (HAM detectado como SPAM):          {cm[0,1]}")
    print(f"  Falsos Negativos (SPAM no detectado):                {cm[1,0]}")
    print(f"  Verdaderos Negativos (HAM correctamente detectado):  {cm[1,1]}")
    print("--------------------------------------------------")
    print(f"Precisión (Exactitud sobre los SPAM detectados):       {precision:.2f}")
    print(f"Recall (Cobertura de los SPAM reales):                 {recall:.2f}")
    print(f"F1-score (Balance entre precisión y recall):           {f1:.2f}")
    print("==================================================")

for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    evaluar_modelo(threshold=t)

def clasificador_interactivo():
    print("\nCLASIFICADOR SPAM/HAM")
    mensaje = input("Ingresa un mensaje de texto: ")
    tokens = preprocess(mensaje)
    prob = P_spam_given_words(tokens)
    clasificado = "SPAM" if prob >= 0.5 else "HAM"
    print(f"\nClasificación del mensaje: {clasificado}")
    print(f"Probabilidad de que sea SPAM: {round(prob * 100, 2)}%")
    palabra_prob = []
    for w in set(tokens):
        PwS = P_w_given_class(w, spam_freq, total_spam_words)
        PwH = P_w_given_class(w, ham_freq, total_ham_words)
        numerador = PwS * P_spam
        denominador = numerador + PwH * P_ham
        if denominador > 0:
            psw = numerador / denominador
            palabra_prob.append((w, psw))
    palabra_prob.sort(key=lambda x: x[1], reverse=True)
    top_palabras = palabra_prob[:3]
    print("Palabras con mayor peso predictivo (más indicativas de SPAM):")
    for palabra, p in top_palabras:
        print(f"- {palabra}: {round(p * 100, 2)}%")

if __name__ == "__main__":
    clasificador_interactivo()
