import datetime
import re
import spacy
import pandas as pd
import streamlit as st
import requests
import urllib.parse
from bs4 import BeautifulSoup
from google_play_scraper import app as gp_app, reviews as gp_reviews, Sort
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

# Настройки API
GOOGLE_PLAY_COUNTRY = "ru"
APP_STORE_COUNTRY = "ru"
MAX_RESULTS = 5

def search_apps(query: str):
    """Поиск приложений по названию"""
    results = {"google_play": [], "app_store": []}
    
    try:
        # Поиск в Google Play
        gp_results = search(
            query,
            lang="ru",
            country=GOOGLE_PLAY_COUNTRY,
            n_hits=MAX_RESULTS
        )
        results["google_play"] = [{
            "id": r["appId"],
            "title": r["title"],
            "developer": r["developer"],
            "score": r["score"]
        } for r in gp_results]
        
    except Exception as e:
        st.error(f"Ошибка поиска в Google Play: {str(e)}")
    
    try:
        # Поиск в App Store через iTunes API
        itunes_response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "country": APP_STORE_COUNTRY,
                "media": "software",
                "limit": MAX_RESULTS
            }
        )
        ios_data = itunes_response.json()
        results["app_store"] = [{
            "id": r["trackId"],
            "title": r["trackName"],
            "developer": r["artistName"],
            "score": r["averageUserRating"]
        } for r in ios_data.get("results", [])]
        
    except Exception as e:
        st.error(f"Ошибка поиска в App Store: {str(e)}")
    
    return results

def display_search_results(results: dict):
    """Отображение результатов поиска"""
    st.subheader("🔍 Результаты поиска")
    
    if not results["google_play"] and not results["app_store"]:
        st.warning("Приложения не найдены")
        return
    
    # Google Play
    if results["google_play"]:
        st.markdown("### Google Play")
        for i, app in enumerate(results["google_play"], 1):
            with st.expander(f"{i}. {app['title']}"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                if st.button(f"Выбрать", key=f"gp_{app['id']}"):
                    st.session_state.selected_gp_app = app
    
    # App Store
    if results["app_store"]:
        st.markdown("### App Store")
        for i, app in enumerate(results["app_store"], 1):
            with st.expander(f"{i}. {app['title']}"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                if st.button(f"Выбрать", key=f"ios_{app['id']}"):
                    st.session_state.selected_ios_app = app

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

def get_google_play_reviews(app_name: str, lang: str = 'ru', country: str = 'ru', count: int = 100) -> tuple:
    app_id = search_google_play(app_name)
    if not app_id:
        return [], 0.0
    
    try:
        app_info = gp_app(app_id, lang=lang, country=country)
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

def get_app_store_reviews(app_name: str, country: str = 'ru', count: int = 100) -> tuple:
    app_id = search_app_store(app_name)
    if not app_id:
        return [], 0.0
    
    try:
        rating = get_app_store_rating(app_id)
        
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
        
        # Фильтрация и нормализация фраз
        filtered_phrases = [
            phrase.strip().lower()
            for phrase in phrases 
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
            
            # Ищем точное вхождение фразы в тексте
            start_idx = text.lower().find(phrase)
            if start_idx != -1:
                # Вырезаем контекст вокруг фразы
                start = max(0, start_idx - 30)
                end = min(len(text), start_idx + len(phrase) + 30)
                example = text[start:end].strip()
                
                # Форматирование примера
                if start > 0:
                    example = "..." + example
                if end < len(text):
                    example += "..."
                    
                example = example.replace(phrase, f"**{phrase}**")
            else:
                example = text[:100] + "..."
            
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(example)
    
    # Фильтруем фразы с одинаковыми примерами
    unique_phrases = []
    for phrase, count in analysis['key_phrases'].most_common():
        unique_examples = list({ex for ex in analysis['examples'][phrase]})
        if len(unique_examples) > 0:
            analysis['examples'][phrase] = unique_examples
            unique_phrases.append((phrase, count))
    
    analysis['key_phrases'] = Counter(dict(unique_phrases[:15]))
    
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
        page_title="Поиск приложений",
        layout="wide",
        menu_items={'About': "### Поиск мобильных приложений v2.0"}
    )
     st.title("📲 Поиск мобильных приложений")
    
    # Поисковая строка
     search_query = st.text_input(
        "Введите название приложения:",
        placeholder="Например: TikTok, СберБанк",
        key="search_input"
    )
    
    # Инициализация состояния
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    
    # Кнопка поиска
    if st.button("🔎 Найти приложения", type="primary"):
        if len(search_query) < 3:
            st.warning("Введите минимум 3 символа для поиска")
        else:
            with st.spinner("Ищем приложения..."):
                st.session_state.search_results = search_apps(search_query)
    
    # Отображение результатов
    if st.session_state.search_results:
        display_search_results(st.session_state.search_results)
    
    # Отображение выбранных приложений
    if 'selected_gp_app' in st.session_state:
        st.success(f"✅ Выбрано приложение Google Play: {st.session_state.selected_gp_app['title']}")
    
    if 'selected_ios_app' in st.session_state:
        st.success(f"✅ Выбрано приложение App Store: {st.session_state.selected_ios_app['title']}")
    
    # Кнопка анализа при наличии выбранных приложений
    if 'selected_gp_app' in st.session_state or 'selected_ios_app' in st.session_state:
        if st.button("🚀 Начать анализ отзывов", type="primary"):
            get_reviews()
            
    if reviews_gp or reviews_ios:
                filtered_reviews = reviews_gp + reviews_ios
                # Фильтрация отзывов по датам
                start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
                end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
                filtered_reviews = filter_reviews_by_date(filtered_reviews, start_datetime, end_datetime)
                
                # Анализируем отзывы
                analysis = analyze_reviews(filtered_reviews)
                
                # Отображаем результаты
                display_analysis(analysis, filtered_reviews)
            else:
                st.warning("Не удалось найти отзывы.")

if __name__ == "__main__":
    main()
