import datetime
import re
import spacy
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter, defaultdict
from transformers import pipeline

# Инициализация NLP только для русского языка
def load_nlp_model():
    try:
        return spacy.load("ru_core_news_sm")
    except OSError:
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

# Модифицированная инициализация модели с кешированием
@st.cache_resource
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "text-classification", 
        model="cointegrated/rubert-tiny-sentiment-balanced",
        framework="pt"
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

def get_app_store_reviews(app_url: str, country: str = 'ru', count: int = 100) -> list:
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

def filter_reviews_by_date(reviews: list, start_date: datetime.datetime, end_date: datetime.datetime) -> list:
    return [r for r in reviews if start_date <= r[0] <= end_date]

def analyze_sentiments(reviews: list) -> list:
    sentiment_analyzer = load_sentiment_model()
    sentiments = []
    for _, text, _ in reviews:
        try:
            result = sentiment_analyzer(text[:512], truncation=True)[0]  # Ограничение длины текста
            sentiments.append({
                'label': result['label'],
                'score': result['score']
            })
        except Exception as e:
            st.error(f"Ошибка анализа: {str(e)}")
            sentiments.append({'label': 'NEUTRAL', 'score': 0.5})
    return sentiments

def extract_key_phrases(text: str) -> list:
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]

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

def display_analysis(analysis: dict, total_reviews: int):
    st.header("📊 Результаты анализа")
    
    cols = st.columns(3)
    cols[0].metric("Всего отзывов", total_reviews)
    cols[1].metric("Google Play", analysis['platform_counts']['Google Play'])
    cols[2].metric("App Store", analysis['platform_counts']['App Store'])
    
    st.subheader("📈 Распределение тональности")
    sentiment_df = pd.DataFrame([
        {'Тональность': 'Позитивные', 'Количество': sum(1 for s in analysis['sentiments'] if s['label'] == 'POSITIVE')},
        {'Тональность': 'Нейтральные', 'Количество': sum(1 for s in analysis['sentiments'] if s['label'] == 'NEUTRAL')},
        {'Тональность': 'Негативные', 'Количество': sum(1 for s in analysis['sentiments'] if s['label'] == 'NEGATIVE')}
    ])
    st.bar_chart(sentiment_df.set_index('Тональность'))
    
    st.subheader("🔑 Ключевые упоминания")
    if analysis['key_phrases']:
        phrases_df = pd.DataFrame(
            analysis['key_phrases'].most_common(10),
            columns=['Фраза', 'Количество']
        )
        st.dataframe(
            phrases_df.style.background_gradient(subset=['Количество'], cmap='Blues'),
            height=400
        )
    else:
        st.info("Ключевые фразы не обнаружены")

def main():
    st.set_page_config(page_title="Анализатор отзывов", layout="wide")
    st.title("📱 Анализатор отзывов приложений")
    
    col1, col2 = st.columns(2)
    with col1:
        gp_url = st.text_input("Ссылка Google Play", placeholder="https://play.google.com/store/apps/details?id=...")
    with col2:
        ios_url = st.text_input("Ссылка App Store", placeholder="https://apps.apple.com/ru/app/...")
    
    start_date = st.date_input("Начальная дата", datetime.date(2024, 1, 1))
    end_date = st.date_input("Конечная дата", datetime.date.today())
    
    if st.button("🔍 Начать анализ", type="primary"):
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
            
        display_analysis(analysis, len(filtered_reviews))

if __name__ == "__main__":
    main()
