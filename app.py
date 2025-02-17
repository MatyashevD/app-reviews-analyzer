import datetime
import re
import spacy
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoap
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
        return [(r['at'], r['content'], 'Google Play', r['score']) for r in result], rating
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
        
        return [(r['date'], r['review'], 'App Store', r['rating']) for r in app.reviews], rating
    except Exception as e:
        st.error(f"Ошибка App Store: {str(e)}")
        return [], 0.0

def filter_reviews_by_date(reviews: list, start_date: datetime.datetime, end_date: datetime.datetime) -> list:
    return [r for r in reviews if start_date <= r[0] <= end_date]

def analyze_sentiments(reviews: list) -> list:
    try:
        sentiment_analyzer = load_sentiment_model()
        return [sentiment_analyzer(text[:512], truncation=True)[0] for _, text, _, _ in reviews]
    except:
        return [{'label': 'neutral', 'score': 0.5} for _ in reviews]

def extract_key_phrases(text: str) -> list:
    try:
        doc = nlp(text)
        phrases = []
        current_phrase = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']:
                current_phrase.append(token.text)
                if len(current_phrase) == 4:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            phrases.append(' '.join(current_phrase))
        
        filtered_phrases = [
            phrase for phrase in phrases 
            if 2 <= len(phrase.split()) <= 4
            and not any(c in phrase for c in ['@', '#', 'http'])
        ]
        
        return list(set(filtered_phrases))
    
    except Exception as e:
        st.error(f"Ошибка обработки текста: {str(e)}")
        return []

def analyze_reviews(reviews: list) -> dict:
    analysis = {
        'sentiments': [],
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'examples': defaultdict(list),
        'gp_rating': 0.0,
        'ios_rating': 0.0
    }
    
    sentiments = analyze_sentiments(reviews)
    
    for idx, (date, text, platform, rating) in enumerate(reviews):
        analysis['sentiments'].append(sentiments[idx])
        analysis['platform_counts'][platform] += 1
        
        phrases = extract_key_phrases(text)
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text[:150] + '...')
    
    return analysis

def display_analysis(analysis: dict, filtered_reviews: list):
    st.header("📊 Результаты анализа")
    
    # Сохраняем состояние
    st.session_state.analysis_data = analysis
    st.session_state.filtered_reviews = filtered_reviews
    
    tab1, tab2 = st.tabs(["Аналитика", "Все отзывы"])
    
    with tab1:
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", len(filtered_reviews))
        cols[1].metric(
            "Google Play", 
            f"{analysis['platform_counts']['Google Play']} отзывов",
            f"★ {analysis['gp_rating']:.1f}" if analysis['gp_rating'] > 0 else ""
        )
        cols[2].metric(
            "App Store", 
            f"{analysis['platform_counts']['App Store']} отзывов",
            f"★ {analysis['ios_rating']:.1f}" if analysis['ios_rating'] > 0 else ""
        )
        
        st.subheader("📈 Распределение тональности")
        sentiment_counts = {
            'Позитивные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'positive'),
            'Нейтральные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'neutral'),
            'Негативные': sum(1 for s in analysis['sentiments'] if s['label'].lower() == 'negative')
        }

        if sum(sentiment_counts.values()) > 0:
            sentiment_df = pd.DataFrame.from_dict(sentiment_counts, orient='index', columns=['Количество'])
            st.bar_chart(sentiment_df)
        else:
            st.warning("Нет данных для отображения тональности")
        
        st.subheader("🔑 Ключевые темы (Топ-15)")
        if analysis['key_phrases']:
            top_phrases = analysis['key_phrases'].most_common(15)
            
            st.markdown("""
            <style>
                .phrase-box {
                    border: 1px solid #e6e6e6;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    background: #f9f9f9;
                }
                .phrase-text {
                    font-weight: 600;
                    color: #2c3e50;
                    font-size: 16px;
                }
                .phrase-count {
                    color: #3498db;
                    font-size: 14px;
                }
                .phrase-example {
                    color: #7f8c8d;
                    font-size: 14px;
                    margin-top: 8px;
                }
            </style>
            """, unsafe_allow_html=True)
            
            for phrase, count in top_phrases:
                examples = analysis['examples'].get(phrase, [])[:2]
                examples_html = "<br>".join([f"• {ex}" for ex in examples])
                
                st.markdown(f"""
                <div class="phrase-box">
                    <div class="phrase-text">
                        {phrase} 
                        <span class="phrase-count">({count} упоминаний)</span>
                    </div>
                    <div class="phrase-example">
                        Примеры:<br>
                        {examples_html if examples else "Нет примеров"}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Ключевые темы не обнаружены")
    
    with tab2:
        st.subheader("📋 Все отзывы")
        sentiment_translation = {
            'positive': 'Позитивный',
            'neutral': 'Нейтральный',
            'negative': 'Негативный'
        }
        
        reviews_df = pd.DataFrame([{
            'Дата': r[0].strftime('%Y-%m-%d'),
            'Платформа': r[2],
            'Оценка': '★' * int(r[3]),
            'Оценка (баллы)': r[3],
            'Отзыв': r[1],
            'Тональность': sentiment_translation.get(s['label'].lower(), 'Нейтральный')
        } for i, (r, s) in enumerate(zip(filtered_reviews, analysis['sentiments']))])
        
        st.dataframe(
            reviews_df[['Дата', 'Платформа', 'Оценка', 'Тональность', 'Отзыв']],
            height=600,
            column_config={
                "Оценка": st.column_config.TextColumn(width="small"),
                "Отзыв": st.column_config.TextColumn(width="large")
            }
        )
        
        csv = reviews_df[['Дата', 'Платформа', 'Оценка (баллы)', 'Тональность', 'Отзыв']]
        csv = csv.to_csv(index=False).encode('utf-8')
        
        if st.download_button(
            label="📥 Скачать все отзывы",
            data=csv,
            file_name='отзывы.csv',
            mime='text/csv',
            key='download_btn'
        ):
            st.session_state.analysis_data = analysis
            st.session_state.filtered_reviews = filtered_reviews
            st.experimental_rerun()
    
    if st.button("🔄 Новый анализ", type="secondary"):
        st.session_state.clear()
        st.experimental_rerun()

def main():
    st.set_page_config(
        page_title="Анализатор отзывов", 
        layout="wide",
        menu_items={'About': "### Анализатор отзывов v4.0"}
    )
    st.title("📱 Анализатор отзывов приложений")
    
    # Сохраняем URL в сессии
    col1, col2 = st.columns(2)
    with col1:
        gp_url = st.text_input(
            "Ссылка Google Play", 
            value=st.session_state.get('gp_url', ''),
            key='gp_url_input'
        )
    with col2:
        ios_url = st.text_input(
            "Ссылка App Store", 
            value=st.session_state.get('ios_url', ''),
            key='ios_url_input'
        )
    
    start_date = st.date_input("Начальная дата", datetime.date(2024, 1, 1))
    end_date = st.date_input("Конечная дата", datetime.date.today())
    
    if st.button("🚀 Начать анализ", type="primary"):
        st.session_state.gp_url = gp_url
        st.session_state.ios_url = ios_url
        
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
            
            with st.spinner("Анализ текста..."):
                analysis = analyze_reviews(filtered_reviews)
                analysis.update({
                    'gp_rating': gp_rating,
                    'ios_rating': ios_rating,
                    'platform_counts': {
                        'Google Play': sum(1 for r in filtered_reviews if r[2] == 'Google Play'),
                        'App Store': sum(1 for r in filtered_reviews if r[2] == 'App Store')
                    }
                })
            
            st.session_state.analysis_data = analysis
            st.session_state.filtered_reviews = filtered_reviews
            st.experimental_rerun()
    
    if 'analysis_data' in st.session_state and 'filtered_reviews' in st.session_state:
        display_analysis(st.session_state.analysis_data, st.session_state.filtered_reviews)

if __name__ == "__main__":
    main()
