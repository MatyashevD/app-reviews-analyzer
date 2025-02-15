import datetime
import re
import spacy
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter, defaultdict
from io import StringIO
from transformers import pipeline

# Инициализация NLP модели
def load_nlp_model():
    try:
        return spacy.load("ru_core_news_sm")
    except:
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

nlp = load_nlp_model()

@st.cache_resource(show_spinner="Загрузка модели анализа тональности...")
def load_sentiment_model():
    return pipeline(
        "text-classification", 
        model="cointegrated/rubert-tiny-sentiment-balanced",
        framework="pt",
        device=-1
    )

def extract_google_play_id(url: str) -> str:
    match = re.search(r'id=([a-zA-Z0-9._-]+)', url)
    return match.group(1) if match else None

def extract_app_store_id(url: str) -> str:
    match = re.search(r'/id(\d+)', url)
    return match.group(1) if match else None

def get_google_play_reviews(app_url: str, lang: str = 'ru', country: str = 'ru', count: int = 100) -> list:
    app_id = extract_google_play_id(app_url)
    if not app_id:
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
    except:
        return []

def get_app_store_reviews(app_url: str, country: str = 'ru', count: int = 100) -> list:
    app_id = extract_app_store_id(app_url)
    if not app_id:
        return []
    
    try:
        app_name_match = re.search(r'/app/([^/]+)/', app_url)
        app_name = app_name_match.group(1) if app_name_match else "unknown_app"
        
        app = AppStore(country=country, app_id=app_id, app_name=app_name)
        app.review(how_many=count)
        
        return [(r['date'], r['review'], 'App Store') for r in app.reviews]
    except:
        return []

def filter_reviews_by_date(reviews: list, start_date: datetime.datetime, end_date: datetime.datetime) -> list:
    return [r for r in reviews if start_date <= r[0] <= end_date]

def analyze_sentiments(reviews: list) -> list:
    try:
        sentiment_analyzer = load_sentiment_model()
        return [sentiment_analyzer(text[:512], truncation=True)[0] for _, text, _ in reviews]
    except:
        return [{'label': 'NEUTRAL', 'score': 0.5} for _ in reviews]

def extract_key_phrases(text: str) -> list:
    try:
        doc = nlp(text)
        phrases = []
        current_phrase = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                current_phrase.append(token.text)
            else:
                if len(current_phrase) > 1:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
        
        if len(current_phrase) > 1:
            phrases.append(' '.join(current_phrase))
            
        return phrases
    except:
        return []

def analyze_reviews(reviews: list) -> dict:
    analysis = {
        'sentiments': [],
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'examples': defaultdict(list)
    }
    
    sentiments = analyze_sentiments(reviews)
    
    for idx, (date, text, platform) in enumerate(reviews):
        analysis['sentiments'].append(sentiments[idx])
        analysis['platform_counts'][platform] += 1
        
        phrases = extract_key_phrases(text)
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text)
    
    return analysis

def display_analysis(analysis: dict, filtered_reviews: list):
    st.header("📊 Результаты анализа")
    
    tab1, tab2 = st.tabs(["Аналитика", "Все отзывы"])
    
    with tab1:
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", len(filtered_reviews))
        cols[1].metric("Google Play", analysis['platform_counts']['Google Play'])
        cols[2].metric("App Store", analysis['platform_counts']['App Store'])
        
        st.subheader("📈 Распределение тональности")
        
        # Добавляем отладочный вывод
        st.write("Примеры меток тональности:", analysis['sentiments'][:3])
        
        # Исправляем ключи согласно выходу модели
        sentiment_counts = {
            'Позитивные': sum(1 for s in analysis['sentiments'] if s['label'].upper() == 'POSITIVE'),
            'Нейтральные': sum(1 for s in analysis['sentiments'] if s['label'].upper() == 'NEUTRAL'),
            'Негативные': sum(1 for s in analysis['sentiments'] if s['label'].upper() == 'NEGATIVE')
        }
        
        if sum(sentiment_counts.values()) > 0:
            sentiment_df = pd.DataFrame.from_dict(sentiment_counts, orient='index', columns=['Количество'])
            st.bar_chart(sentiment_df)
        else:
            st.warning(f"Нет данных для отображения. Всего записей: {len(analysis['sentiments']}")
        
        st.subheader("🔑 Ключевые упоминания")
        if analysis['key_phrases']:
            phrases_df = pd.DataFrame(
                analysis['key_phrases'].most_common(10),
                columns=['Фраза', 'Количество']
            )
            st.dataframe(phrases_df, height=400)
        else:
            st.info("Ключевые фразы не обнаружены")
    
    with tab2:
        st.subheader("📋 Все отзывы")
        reviews_df = pd.DataFrame([{
            'Дата': r[0].strftime('%Y-%m-%d'),
            'Платформа': r[2],
            'Отзыв': r[1],
            'Тональность': analysis['sentiments'][i]['label']
        } for i, r in enumerate(filtered_reviews)])
        
        st.dataframe(reviews_df, height=500)
        
        csv = reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Скачать CSV",
            data=csv,
            file_name='reviews.csv',
            mime='text/csv'
        )

def main():
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")
    st.title("📱 Анализатор отзывов приложений")
    
    col1, col2 = st.columns(2)
    with col1:
        gp_url = st.text_input("Google Play URL", placeholder="https://play.google.com/store/apps/details?id=...")
    with col2:
        ios_url = st.text_input("App Store URL", placeholder="https://apps.apple.com/ru/app/...")
    
    start_date = st.date_input("Начальная дата", datetime.date(2024, 1, 1))
    end_date = st.date_input("Конечная дата", datetime.date.today())
    
    if st.button("🚀 Начать анализ"):
        with st.spinner("Сбор отзывов..."):
            gp_revs = get_google_play_reviews(gp_url)
            ios_revs = get_app_store_reviews(ios_url)
            all_reviews = gp_revs + ios_revs
            
            if not all_reviews:
                st.error("Отзывы не найдены!")
                return
                
            start_dt = datetime.datetime.combine(start_date, datetime.time.min)
            end_dt = datetime.datetime.combine(end_date, datetime.time.max)
            filtered_reviews = filter_reviews_by_date(all_reviews, start_dt, end_dt)
            
            platform_counts = {
                'Google Play': sum(1 for r in filtered_reviews if r[2] == 'Google Play'),
                'App Store': sum(1 for r in filtered_reviews if r[2] == 'App Store')
            }
            
        with st.spinner("Анализ текста..."):
            analysis = analyze_reviews(filtered_reviews)
            analysis['platform_counts'] = platform_counts
            
        display_analysis(analysis, filtered_reviews)

if __name__ == "__main__":
    main()
