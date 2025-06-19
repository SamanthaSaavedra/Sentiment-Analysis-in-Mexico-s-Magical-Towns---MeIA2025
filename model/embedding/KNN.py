import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')

# --- Parámetro modificable ---
rare_words_percentage = 1  # 10% de palabras menos frecuentes

# --- 1. Cargar y preparar dataset ---
df = pd.read_excel("dataset/MeIA_2025_train.xlsx")
df["labels"] = df["Polarity"] - 1
df = df[["Review", "labels"]]
df["labels"] = df["labels"].astype(int)

# --- 2. Preprocesamiento simple ---
stop_words = set(stopwords.words("spanish"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

df["tokens"] = df["Review"].apply(clean_text)

# --- 3. Seleccionar solo el 10% de palabras menos frecuentes ---
all_tokens = [token for tokens in df["tokens"] for token in tokens]
word_freq = Counter(all_tokens)
sorted_words = sorted(word_freq.items(), key=lambda x: x[1])  # orden creciente

num_rare_words = int(len(sorted_words) * rare_words_percentage)
rare_words = set([word for word, freq in sorted_words[:num_rare_words]])

# Filtrar tokens por rare_words
def filter_rare(tokens):
    return [t for t in tokens if t in rare_words]

df["tokens"] = df["tokens"].apply(filter_rare)

# --- 4. Entrenar modelo Word2Vec solo con rare words ---
sentences = df["tokens"].tolist()
w2v_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=1)

# --- 5. Convertir texto a vector promedio ---
def text_vector(tokens):
    vecs = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
    if len(vecs) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(vecs, axis=0)

df["vector"] = df["tokens"].apply(text_vector)

# --- 6. Separar en entrenamiento y validación ---
X = np.stack(df["vector"].values)
y = df["labels"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 7. Evaluar múltiples valores de k ---
best_k = None
best_acc = 0
results = {}

print("\nEvaluando KNN con palabras menos frecuentes (10%)...\n")
for k in range(1, 1000):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_val_pred = knn.predict(X_val)

    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    results[k] = val_acc

    print(f"k = {k:2d} | Train Accuracy = {train_acc:.4f} | Validation Accuracy = {val_acc:.4f}")

    if val_acc > best_acc:
        best_k = k
        best_acc = val_acc

# --- 8. Resultado óptimo ---
print(f"\n🔍 Mejor k encontrado: {best_k} con Validation Accuracy = {best_acc:.4f}")