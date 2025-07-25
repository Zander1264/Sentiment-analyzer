# Импортируем необходимые библиотеки
import sys
import os
import pathlib
import networkx as nx
import matplotlib.pyplot as plt
import csv
import pymorphy3 as pymorphy3
import pymorphy3_dicts_ru
import nltk
import numpy as np

if getattr (sys, 'frozen', False) and hasattr (sys, '_MEIPASS'):
    os.environ ["PYMORPHY3_DICT_PATH"] = str (pathlib.Path (sys._MEIPASS).joinpath ('data'))

morph = pymorphy3.MorphAnalyzer(lang='ru')

# Загружаем данные о тональности слов и выражений из локальных файлов ksrtaslovsent.csv
# Формат файла: слово или выражение, тональность от -1 (отрицательная) до 1 (положительная)
# Открываем файл csv в режиме чтения
with open("kartaslovsent.csv", "r", encoding='utf-8') as f:
    # Создаем объект csv.reader для чтения данных из файла
    reader = csv.reader(f)
    # Пропускаем первую строку, если она содержит заголовки
    next(reader)
    # Создаем пустой словарь тональности
    sentiment_dict = {}
    # Для каждой строки в файле
    for row in reader:
        # Получаем слово и тональность из строки
        word, sentiment = row
        # Преобразуем тональность в число с плавающей точкой
        sentiment = float(sentiment)
        # Добавляем пару слово-тональность в словарь
        sentiment_dict[word] = sentiment


# Функция для предобработки текста
def preprocess_text(text):
    # Разбиваем текст на предложения
    sentences = nltk.sent_tokenize(text, language="russian")
    # Разбиваем каждое предложение на слова
    words = [nltk.word_tokenize(sentence, language="russian") for sentence in sentences]
    # Удаляем знаки препинания и стоп-слова
    punctuation = ".,:;!?()''\"\'\"\\``-"
    stopwords = nltk.corpus.stopwords.words("russian")
    words = [[word.lower() for word in sentence if word not in punctuation and word not in stopwords] for sentence in words]
    # Приводим слова к нормальной форме
    lemmatizer = pymorphy3.MorphAnalyzer()
    words = [[lemmatizer.parse(word)[0].normal_form for word in sentence] for sentence in words]
    return words


# Функция для определения тональности слова или выражения
def get_sentiment(word):
    # Если слово или выражение есть в словаре тональности, возвращаем его значение
    if word in sentiment_dict:
        return sentiment_dict[word]
    # Иначе возвращаем 0 (нейтральная тональность)
    else:
        return 0


# Функция для построения семантической сети
def build_semantic_network(words):
    # Создаем пустую сеть
    network = nx.Graph()
    # Для каждого предложения в тексте
    for sentence in words:
        # Для каждого слова или выражения в предложении
        for word in sentence:
            # Добавляем вершину в сеть с атрибутом тональности
            network.add_node(word, sentiment=get_sentiment(word))
            # Для каждого другого слова или выражения в предложении
            for other_word in sentence:
                # Если это не то же самое слово или выражение
                if word != other_word:
                    # Добавляем ребро в сеть с весом, равным произведению тональностей
                    network.add_edge(word, other_word, weight=get_sentiment(word) * get_sentiment(other_word))
    return network


# Функция для определения тональности текста или предложения
def get_text_sentiment(network, words):
    # Создаем пустой список для хранения тональностей предложений
    sentence_sentiments = []
    # Для каждого предложения в тексте
    for sentence in words:
        # Создаем пустой список для хранения тональностей слов или выражений в предложении
        word_sentiments = []
        # Для каждого слова или выражения в предложении
        for word in sentence:
            # Добавляем его тональность в список
            word_sentiments.append(get_sentiment(word))
        # Вычисляем среднюю тональность предложения
        sentence_sentiment = sum(word_sentiments) / len(word_sentiments)
        # Добавляем ее в список
        sentence_sentiments.append(sentence_sentiment)
    # Вычисляем среднюю тональность текста
    text_sentiment = sum(sentence_sentiments) / len(sentence_sentiments)
    return text_sentiment, sentence_sentiments


# Функция для вывода результатов анализа
def show_results(text, network, text_sentiment, sentence_sentiments):
    # Выводим текст
    print("Текст для анализа:")
    print(text)
    print()
    # Выводим общую тональность текста
    print("Общий эмоциональный окрас текста: {:.2f}".format(text_sentiment))
    # Выводим тональность каждого предложения
    sentences = nltk.sent_tokenize(text, language="russian")
    for i, sentence in enumerate(sentences):
        print("Эмоциональный окрас предложения {}: {:.2f}".format(i + 1, sentence_sentiments[i]))
    print()
    # Выводим семантическую сеть
    print("Семантическая сеть:")
    # Определяем цвета вершин в зависимости от их тональности
    colors = []
    # Создаем словарь, который хранит номера предложений для каждого слова
    sentence_numbers = {}
    # Получаем список нормализованных слов из текста
    normalized_words = preprocess_text(text)
    for node in network.nodes():
        sentiment = network.nodes[node]["sentiment"]
        if sentiment > 0:
            colors.append("green")
        elif sentiment < 0:
            colors.append("red")
        else:
            colors.append("grey")
        # Сохраняем цвет вершины в атрибутах узла
        network.nodes[node]["color"] = colors[-1]
        # Находим номер предложения, в котором встречается нормализованное слово
        for i, sentence in enumerate(normalized_words):
            if node in sentence:
                sentence_numbers[node] = i + 1
                break
    # Рисуем сеть с помощью библиотеки matplotlib
    # Определяем количество строк и столбцов в сетке графиков
    # В зависимости от количества предложений
    n = len(sentences)
    # Создаем фигуру
    fig = plt.figure(figsize=(10, 10))
    # Настраиваем расстояние между графиками
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    # Создаем и отрисовываем подграфы для каждого предложения
    for i, sentence in enumerate(sentences):
        # Выбираем только те слова, которые принадлежат i-му предложению
        subgraph = network.subgraph([word for word in network.nodes() if sentence_numbers[word] == i + 1])
        # Определяем цвета вершин в зависимости от их тональности
        colors = [network.nodes[word]["color"] for word in subgraph.nodes()]
        # Добавляем подграф в фигуру
        ax = fig.add_subplot(n // 3 + 1, 3, i + 1)
        # Рисуем подграф
        nx.draw_networkx(subgraph, ax=ax, with_labels=True, node_color=colors, font_size=12,
                         node_size=100, edge_color="blue")
        ax.set_title("График предложения {}".format(i + 1))
    plt.show()

if __name__ == "__main__":

    # Считываем текст из файла или запрашиваем у пользователя ввод текста
    text = input("Введите текст для анализа или оставьте поле пустым, чтобы считать текст из файла: ")
    if not text:
        with open("text.txt", encoding="utf-8") as f:
            text = f.read()

    # Предобрабатываем текст
    words = preprocess_text(text)

    # Строим семантическую сеть
    network = build_semantic_network(words)

    # Определяем тональность текста или предложений
    text_sentiment, sentence_sentiments = get_text_sentiment(network, words)

    # Выводим результаты анализа
    show_results(text, network, text_sentiment, sentence_sentiments)

    input("Для выхода нажмите Enter: ")
