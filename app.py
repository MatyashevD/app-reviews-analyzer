import datetime
import spacy
import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter

nlp = spacy.load("ru_core_news_sm")

def extract_google_play_id(url):
    match = re.search(r'id=([a-zA-Z0-9._-]+)', url)
    return match.group(1) if match else None

def extract_app_store_id(url):
    match = re.search(r'/id(\d+)', url)
    return match.group(1) if match else None

def get_google_play_reviews(app_url, lang='ru', country='ru', count=100):
    app_id = extract_google_play_id(app_url)
    if not app_id:
        st.warning("Неверный URL Google Play")
        return []
    
    try:
        result, _ = gp_reviews(
            app_id,
            lang=lang,
            country=country,
            count=count,
            sort=Sort.NEWEST
        )
        return [(r['at'], r['content'], 'Google Play') for r in result]
    except Exception as e:
        st.error(f"Ошибка Google Play: {str(e)}")
        return []

def get_app_store_reviews(app_url, country='ru', count=100):
    app_id = extract_app_store_id(app_url)
    if not app_id:
        st.warning("Неверный URL App Store")
        return []
    
    try:
        app_name_match = re.search(r'/app/([^/]+)/', app_url)
        app_name = app_name_match.group(1) if app_name_match else "unknown_app"
        
        app = AppStore(country=country, app_id=app_id, app_name=app_name)
        app.review(how_many=count)
        
        return [(r['date'], r['review'], 'App Store') for r in app.reviews]
    except Exception as e:
        st.error(f"Ошибка App Store: {str(e)}")
        return []

def filter_reviews_by_date(reviews, start_date, end_date):
    return [r for r in reviews if start_date <= r[0] <= end_date]

def extract_top_mentions(reviews, top_n=15, examples_per_word=5):
    word_stats = {}
    
    for date, text, platform in reviews:
        doc = nlp(text.lower())
        for token in doc:
            if token.is_alpha and not token.is_stop:
                word = token.text
                if word not in word_stats:
                    word_stats[word] = {
                        'total': 0,
                        'Google Play': 0,
                        'App Store': 0,
                        'examples': [],
                        'dates': []
                    }
                word_stats[word]['total'] += 1
                word_stats[word][platform] += 1
                
                if len(word_stats[word]['examples']) < examples_per_word:
                    word_stats[word]['examples'].append(text)
                    word_stats[word]['dates'].append(date)
    
    return sorted([
        {
            'Упоминание': word,
            'Всего': stats['total'],
            'Google Play': stats['Google Play'],
            'App Store': stats['App Store'],
            'Примеры': stats['examples'],
            'Даты': stats['dates']
        } for word, stats in word_stats.items()
    ], key=lambda x: x['Всего'], reverse=True)[:top_n]

def group_by_topics(reviews):
    topics = {
        'Технические проблемы': ['глючит', 'вылетает', 'тормозит', 'баг', 'ошибка'],
        'Удобство использования': ['интерфейс', 'удобно', 'неудобно', 'дизайн', 'навигация'],
        'Оплата/Доставка': ['оплата', 'доставка', 'курьер', 'цена', 'стоимость'],
        'Безопасность': ['безопасность', 'данные', 'пароль', 'аккаунт', 'взлом'],
        'Обновления': ['обновление', 'версия', 'исправлено', 'новая', 'старая']
    }
    
    topic_stats = {topic: {'count': 0, 'words': []} for topic in topics}
    
    for _, text, _ in reviews:
        doc = nlp(text.lower())
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]
        
        for topic, keywords in topics.items():
            matches = [word for word in words if word in keywords]
            if matches:
                topic_stats[topic]['count'] += 1
                topic_stats[topic]['words'].extend(matches)
    
    return sorted([
        {
            'Тема': topic,
            'Упоминания': stats['count'],
            'Ключевые слова': ', '.join([f"{w[0]} ({w[1]})" for w in Counter(stats['words']).most_common(3)])
        } for topic, stats in topic_stats.items() if stats['count'] > 0
    ], key=lambda x: x['Упоминания'], reverse=True)

def perform_analysis(gp_url, ios_url, start_date, end_date):
    with st.spinner("Сбор отзывов..."):
        gp_revs = get_google_play_reviews(gp_url)
        ios_revs = get_app_store_reviews(ios_url)
    
    all_reviews = gp_revs + ios_revs
    if not all_reviews:
        st.error("Отзывы не найдены!")
        return None
    
    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt = datetime.datetime.combine(end_date, datetime.time.max)
    filtered_reviews = filter_reviews_by_date(all_reviews, start_dt, end_dt)
    filtered_gp = [r for r in filtered_reviews if r[2] == 'Google Play']
    filtered_ios = [r for r in filtered_reviews if r[2] == 'App Store']
    
    with st.spinner("Анализ текста..."):
        analysis_data = {
            'mentions': extract_top_mentions(filtered_reviews),
            'topics': group_by_topics(filtered_reviews),
            'filtered_reviews': filtered_reviews,
            'gp_count': len(filtered_gp),
            'ios_count': len(filtered_ios)
        }
    
    return analysis_data

def main():
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")
    st.title("📱 Анализатор отзывов приложений")
    
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'selected_word' not in st.session_state:
        st.session_state.selected_word = None
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Топ упоминаний"

    col1
::contentReference[oaicite:0]{index=0}
 
