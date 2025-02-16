import datetime
import re
import spacy
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter, defaultdict
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

def get_google_play_reviews(app_url: str, lang: str = 'ru', country: str = 'ru', count: int = 100) -> tuple:
    app_id = extract_google_play_id(app_url)
    if not app_id:
        return [], 0.0
    
    try:
        from google_play_scraper import app
        app_info = app(app_id, lang=lang, country=country)
        rating = app_info.get('score', 0.0)
        
        result, _ = gp_reviews(
            app_id,
            lang=lang,
            country=country,
            count=count,
            sort=Sort.NEWEST
        )
        return [(r['at'], r['content'], 'Google Play') for r in result], rating
    except:
        return [], 0.0

def get_app_store_reviews(app_url: str, country: str = 'ru', count: int = 100) -> tuple:
    app_id = extract_app_store_id(app_url)
    if not app_id:
        return [], 0.0
    
    try:
        rating = get_app_store_rating(app_id)
        
        app_name_match = re.search(r'/app/([^/]+)/', app_url)
        app_name = app_name_match.group(1) if app_name_match else "unknown_app"
        
        app = AppStore(country=country, app_id=app_id, app_name=app_name)
        app.review(how_many=count)
        
        return [(r['date'], r['review'], 'App Store') for r in app.reviews], rating
    except Exception as e:
        st.error(f"Ошибка App Store: {str(e)}")
        return [], 0.0

        # Получаем рейтинг через парсинг страницы
        def get_app_store_rating(app_id: str) -> float:
        try:
            url = f"https://itunes.apple.com/ru/lookup?id={app_id}"
            response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        })
            data = response.json()
        return float(data['results'][0]['averageUserRating']) if data.get('results') else 0.0
    except Exception as e:
            st.error(f"Ошибка получения рейтинга App Store: {str(e)}")
        return 0.0

        rating = get_app_store_rating(app_id)
        
        app_name_match = re.search(r'/app/([^/]+)/', app_url)
        app_name = app_name_match.group(1) if app_name_match else "unknown_app"
        
        app = AppStore(country=country, app_id=app_id, app_name=app_name)
        app.review(how_many=count)
        
        return [(r['date'], r['review'], 'App Store') for r in app.reviews], rating
    except Exception as e:
        st.error(f"Ошибка App Store: {str(e)}")
        return [], 0.0

def filter_reviews_by_date(reviews: list, start_date: datetime.datetime, end_date: datetime.datetime) -> list:
    return [r for r in reviews if start_date <= r[0] <= end_date]

def analyze_sentiments(reviews: list) -> list:
    try:
        sentiment_analyzer = load_sentiment_model()
        return [sentiment_analyzer(text[:512], truncation=True)[0] for _, text, _ in reviews]
    except:
        return [{'label': 'neutral', 'score': 0.5} for _ in reviews]

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
            if len(analysis['examples'][phrase]) < 10:
                analysis['examples'][phrase].append(text)
    
    return analysis

def display_analysis(analysis: dict, filtered_reviews: list):
    st.header("📊 Результаты анализа")
    
    tab1, tab2 = st.tabs(["Аналитика", "Все отзывы"])
    
    with tab1:
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", len(filtered_reviews))
        cols[1].metric(
            "Google Play", 
            f"{analysis['platform_counts']['Google Play']['count']} отзывов",
            f"★ {analysis['platform_counts']['Google Play']['rating']:.1f}"
        )
        cols[2].metric(
            "App Store", 
            f"{analysis['platform_counts']['App Store']['count']} отзывов",
            f"★ {analysis['platform_counts']['App Store']['rating']:.1f}"
        
        st.subheader("📈 Распределение тональности")
        
        # Перевод и подсчет меток
        sentiment_translation = {
            'positive': 'Позитивные',
            'neutral': 'Нейтральные',
            'negative': 'Негативные'
        }
        
        sentiment_counts = {
            'Позитивные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'positive'),
            'Нейтральные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'neutral'),
            'Негативные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'negative')
        }

        if sum(sentiment_counts.values()) > 0:
            # Визуализация
            sentiment_df = pd.DataFrame.from_dict(sentiment_counts, orient='index', columns=['Количество'])
            st.bar_chart(sentiment_df)
            
            # Расчет средних оценок
            pos_scores = [s['score'] for s in analysis['sentiments'] if s['label'].lower() == 'positive']
            neg_scores = [s['score'] for s in analysis['sentiments'] if s['label'].lower() == 'negative']
            
            avg_scores = {
                'Позитивные': round(sum(pos_scores)/len(pos_scores), 3) if pos_scores else 0,
                'Негативные': round(sum(neg_scores)/len(neg_scores), 3) if neg_scores else 0
            }
            
            # Отображение метрик
            st.write("**Точность анализа:**")
            cols = st.columns(2)
            cols[0].metric("Средняя уверенность в позитивных", f"{avg_scores['Позитивные']:.2f}")
            cols[1].metric("Средняя уверенность в негативных", f"{avg_scores['Негативные']:.2f}")
        else:
            st.warning("Нет данных для отображения тональности")
        
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
    
    with tab2:
        st.subheader("📋 Все отзывы")
        # Создаем DataFrame с переведенными метками
        sentiment_translation = {
            'positive': 'Позитивный',
            'neutral': 'Нейтральный',
            'negative': 'Негативный'
        }
        
        reviews_df = pd.DataFrame([{
            'Дата': r[0].strftime('%Y-%m-%d'),
            'Платформа': r[2],
            'Отзыв': r[1],
            'Тональность': sentiment_translation.get(s['label'].lower(), 'Нейтральный')
        } for i, (r, s) in enumerate(zip(filtered_reviews, analysis['sentiments']))])
        
        st.dataframe(reviews_df, height=500, use_container_width=True)
        
        # Экспорт в CSV
        csv = reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать все отзывы",
            data=csv,
            file_name='отзывы.csv',
            mime='text/csv',
            help="Скачать все отзывы в формате CSV"
        )

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
    
    if st.button("🚀 Начать анализ", type="primary"):
        with st.spinner("Сбор отзывов..."):
            gp_revs, gp_rating = get_google_play_reviews(gp_url)
            ios_revs, ios_rating = get_app_store_reviews(ios_url)
            all_reviews = gp_revs + ios_revs
            
            if not all_reviews:
                st.error("Отзывы не найдены!")
                return
                
            start_dt = datetime.datetime.combine(start_date, datetime.time.min)
            end_dt = datetime.datetime.combine(end_date, datetime.time.max)
            filtered_reviews = filter_reviews_by_date(all_reviews, start_dt, end_dt)
            
            platform_counts = {
                'Google Play': {
                    'count': sum(1 for r in filtered_reviews if r[2] == 'Google Play'),
                    'rating': gp_rating
                },
                'App Store': {
                    'count': sum(1 for r in filtered_reviews if r[2] == 'App Store'),
                    'rating': ios_rating
                }
            }
            
        with st.spinner("Анализ текста..."):
            analysis = analyze_reviews(filtered_reviews)
            analysis['platform_counts'] = platform_counts
            
        display_analysis(analysis, filtered_reviews)

if __name__ == "__main__":
    main()
