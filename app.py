import json
import re
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Загружаем JSON-данные из файла
with open('C:/Users/user/Documents/chatbot/faq.json', 'r', encoding='utf-8') as file:   
    faq_data = json.load(file)

# Предобработка вопросов и ответов
questions = []
answers = []

for item in faq_data['faq']:
    for question in item['questions']:
        questions.append(question)
        answers.append(item['answer'])

# Создаем векторизатор TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def normalize_input(user_input):
    # Приводим к нижнему регистру и убираем лишние пробелы
    return re.sub(r'\s+', ' ', user_input.strip().lower())

def get_response(user_input):
    normalized_input = normalize_input(user_input)
    user_tfidf = vectorizer.transform([normalized_input])
    
    # Вычисляем косинусное сходство
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Находим индекс вопроса с максимальным сходством
    highest_index = np.argmax(cosine_similarities)
    highest_score = cosine_similarities[highest_index]

    if highest_score >= 0.6:  # Порог схожести
        return answers[highest_index]
    return "Извините, я не понимаю ваш вопрос. Можете переформулировать?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('question', '')
    response = get_response(user_input)
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(debug=True)