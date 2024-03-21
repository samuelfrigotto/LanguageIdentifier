from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# se precisar baixar 
#nltk.download('stopwords')
#nltk.download('punkt')

def load_data_nltk(file_path):
    textos = []
    labels = []

    # Obtemos o rótulo do idioma a partir do nome do arquivo
    language_label = os.path.basename(file_path).split('_')[1].split('.')[0]

    with open(file_path, 'r', encoding='latin-1') as file:
        lines = file.readlines()
        for line in lines:
            textos.append(line.strip())
            labels.append(language_label)

    return textos, labels

def preprocess_text(text, lang):

    # Remover Pontuações
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenização
    words = word_tokenize(text)
    
    # Remover Stopwords
    if lang != 'unknown':
        stop_words = set(stopwords.words(lang))
        words = [word for word in words if word.lower() not in stop_words]
    
    # Reconstruir texto após pré-processamento
    processed_text = ' '.join(words)
    
    return processed_text


data_path = "D:\Codes\Sistemas Inteligentes" # Local do arquivo.py e da base de dados

##Carregar Base de dados do livro Moby Dick
textos_en, labels_en = load_data_nltk(os.path.join(data_path, "MOBY_EN.txt"))
textos_fr, labels_fr = load_data_nltk(os.path.join(data_path, "MOBY_FR.txt"))
textos_es, labels_es = load_data_nltk(os.path.join(data_path, "MOBY_ES.txt"))
textos_pt, labels_pt = load_data_nltk(os.path.join(data_path, "MOBY_PT.txt"))
textos_de, labels_de = load_data_nltk(os.path.join(data_path, "MOBY_DE.txt"))

##Carregar Base de dados da Wikipedia
textos_fr2, labels_fr2 = load_data_nltk(os.path.join(data_path, "wiki_fr.txt"))
textos_en2, labels_en2 = load_data_nltk(os.path.join(data_path, "wiki_en.txt"))
textos_es2, labels_es2 = load_data_nltk(os.path.join(data_path, "wiki_es.txt"))
textos_pt2, labels_pt2 = load_data_nltk(os.path.join(data_path, "wiki_pt.txt"))
textos_de2, labels_de2 = load_data_nltk(os.path.join(data_path, "wiki_de.txt"))

##Juntar as bases de dados para o treinamento
textos_en += textos_en2
labels_en += labels_en2

textos_fr += textos_fr2
labels_fr += labels_fr2

textos_es += textos_es2
labels_es += labels_es2

textos_pt += textos_pt2
labels_pt += labels_pt2

textos_de += textos_de2
labels_de += labels_de2


en = 'english'
fr = 'french'
es = 'spanish'
pt = 'portuguese'
de = 'german'


##Pre-processamento dos dados
textos_en = [preprocess_text(text, en) for text in textos_en]
textos_fr = [preprocess_text(text, fr) for text in textos_fr]
textos_es = [preprocess_text(text, es) for text in textos_es]
textos_pt = [preprocess_text(text, pt) for text in textos_pt]
textos_de = [preprocess_text(text, de) for text in textos_de]

# Concatenar textos e rótulos
textos = textos_en + textos_fr + textos_es + textos_de + textos_pt
labels = labels_en + labels_fr + labels_es + labels_de + labels_pt

# Vetorização dos dados
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(textos)

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Treinamento do modelo
model = MultinomialNB() #Naive Bayes
model.fit(X_train, y_train)

# Avaliação do modelo
y_pred = model.predict(X_test)

# Exemplo de uso do modelo treinado
while True:
    print("\n\n================   DETECTOR DE IDIOMAS   ================\n\n")
    new_text = input("Digite o texto: ")
    if new_text == 'exit': exit() ## DIGITE "exit" PARA SAIR DO PROGRAMA
    new_text_vectorized = vectorizer.transform([preprocess_text(new_text, 'unknown')])
    predicted_language = model.predict(new_text_vectorized)[0]
    print(f"Idioma previsto para o novo texto: {predicted_language}")
    
