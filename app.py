import datetime
import re
import streamlit as st
import requests
import pandas as pd
from google_play_scraper import search, app, reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import defaultdict, Counter
import spacy
from fuzzywuzzy import fuzz
from itertools import groupby

# Инициализация NLP модели
def load_nlp_model():
    try:
        return spacy.load("ru_core_news_sm")
    except:
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

nlp = load_nlp_model()

# Настройки поиска
MAX_RESULTS = 5
DEFAULT_LANG = 'ru'
DEFAULT_COUNTRY = 'ru'

def search_apps(query: str):
    """Улучшенный поиск приложений с нечетким соответствием"""
    results = {"google_play": [], "app_store": []}
    
    try:
        # Поиск в Google Play
        gp_results = search(
            query,
            lang=DEFAULT_LANG,
            country=DEFAULT_COUNTRY,
            n_hits=MAX_RESULTS
        )
        results["google_play"] = [{
            "id": r["appId"],
            "title": r["title"],
            "developer": r["developer"],
            "score": r["score"],
            "url": f"https://play.google.com/store/apps/details?id={r['appId']}"
        } for r in gp_results]
        
    except Exception as e:
        st.error(f"Ошибка поиска в Google Play: {str(e)}")
    
    try:
        # Поиск в App Store через iTunes API
        itunes_response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "country": DEFAULT_COUNTRY,
                "media": "software",
                "limit": 20,
                "entity": "software,iPadSoftware",
                "lang": "ru_ru"
            },
            headers={"User-Agent": "Mozilla/5.0"}
        )
        ios_data = itunes_response.json()
        
        # Обработка результатов с нечетким поиском
        sorted_results = sorted(ios_data.get("results", []), 
                              key=lambda x: x['trackName'])
        grouped = groupby(sorted_results, key=lambda x: x['trackName'])
        
        processed = []
        for name, group in grouped:
            best_match = max(group, 
                           key=lambda x: fuzz.token_set_ratio(query, x['trackName']))
            processed.append(best_match)

        processed.sort(key=lambda x: fuzz.token_set_ratio(query, x['trackName']), 
                      reverse=True)
        
        results["app_store"] = [{
            "id": r["trackId"],
            "title": r["trackName"],
            "developer": r["artistName"],
            "score": r.get("averageUserRating", 0),
            "url": r["trackViewUrl"],
            "match_score": fuzz.token_set_ratio(query, r['trackName'])
        } for r in processed if r.get('averageUserRating', 0) > 0][:MAX_RESULTS]
        
    except Exception as e:
        st.error(f"Ошибка поиска в App Store: {str(e)}")
    
    return results

def display_search_results(results: dict):
    """Отображение результатов поиска"""
    st.subheader("🔍 Результаты поиска")
    
    if not results["google_play"] and not results["app_store"]:
        st.warning("Приложения не найдены")
        return
    
    if results["google_play"]:
        st.markdown("### Google Play")
        for i, app in enumerate(results["google_play"], 1):
            with st.expander(f"{i}. {app['title']}"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                st.write(f"**Ссылка:** {app['url']}")
                if st.button(f"Выбрать", key=f"gp_{app['id']}"):
                    st.session_state.selected_gp_app = app
    
    if results["app_store"]:
        st.markdown("### App Store")
        for i, app in enumerate(results["app_store"], 1):
            with st.expander(f"{i}. {app['title']} ({app['match_score']}% совпадение)"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                st.write(f"**Ссылка:** {app['url']}")
                if st.button(f"Выбрать", key=f"ios_{app['id']}"):
                    st.session_state.selected_ios_app = app

def get_reviews(app_id: str, platform: str):
    """Получение отзывов с обработкой ошибок"""
    try:
        if platform == 'google_play':
            result, _ = gp_reviews(
                app_id,
                lang=DEFAULT_LANG,
                country=DEFAULT_COUNTRY,
                count=100,
                sort=Sort.NEWEST
            )
            return [(r['at'], r['content'], 'Google Play', r['score']) for r in result]
        else:
            app_store_app = AppStore(
                country=DEFAULT_COUNTRY, 
                app_id=app_id, 
                app_name=st.session_state.selected_ios_app['title']
            )
            app_store_app.review(how_many=100)
            return [(r['date'], r['review'], 'App Store', r['rating']) for r in app_store_app.reviews]
    except Exception as e:
        st.error(f"Ошибка получения отзывов: {str(e)}")
        return []

def extract_key_phrases(text: str) -> list:
    """Улучшенное извлечение фраз для русского языка"""
    try:
        doc = nlp(text)
        phrases = []
        current_phrase = []
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop:
                current_phrase.append(token.text)
                if len(current_phrase) == 3:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
            else:
                if current_phrase:
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            phrases.append(' '.join(current_phrase))
        
        # Фильтрация результатов
        return [
            phrase.strip().lower()
            for phrase in phrases
            if 2 <= len(phrase.split()) <= 3
            and len(phrase) > 4
        ]
    except Exception as e:
        st.error(f"Ошибка обработки текста: {str(e)}")
        return []

def analyze_reviews(filtered_reviews: list):
    """Анализ отзывов с улучшенной обработкой"""
    analysis = {
        'sentiments': [],
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'examples': defaultdict(list),
        'total_reviews': len(filtered_reviews)
    }
    
    for idx, (date, text, platform, rating) in enumerate(filtered_reviews):
        analysis['platform_counts'][platform] += 1
        
        phrases = extract_key_phrases(text)
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text[:100] + '...')
    
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
        page_title="Анализатор приложений",
        layout="wide",
        menu_items={'About': "### Анализатор мобильных приложений v4.0"}
    )
    st.title("📱 Анализатор мобильных приложений")
    
    # Инициализация состояния
    session_defaults = {
        'search_results': None,
        'selected_gp_app': None,
        'selected_ios_app': None,
        'analysis_data': None,
        'filtered_reviews': []
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Поисковая строка
    search_query = st.text_input(
        "Введите название приложения:",
        placeholder="Например: СберБанк, Авито",
        key="search_input"
    )
    
    # Кнопка поиска
    if st.button("🔎 Найти приложения", type="primary"):
        if len(search_query) < 3:
            st.warning("Введите минимум 3 символа для поиска")
        else:
            with st.spinner("Ищем приложения..."):
                st.session_state.search_results = search_apps(search_query)
    
    # Отображение результатов поиска
    if st.session_state.search_results:
        display_search_results(st.session_state.search_results)
    
    # Управление выбором приложений
    selected_apps = []
    if st.session_state.selected_gp_app:
        selected_apps.append(f"Google Play: {st.session_state.selected_gp_app['title']}")
    if st.session_state.selected_ios_app:
        selected_apps.append(f"App Store: {st.session_state.selected_ios_app['title']}")
    
    if selected_apps:
        st.success("✅ Выбрано: " + " | ".join(selected_apps))
    
    # Кнопка анализа
    if selected_apps and st.button("🚀 Начать анализ отзывов", type="primary"):
        with st.spinner("Сбор данных..."):
            all_reviews = []
            
            if st.session_state.selected_gp_app:
                gp_revs = get_reviews(
                    st.session_state.selected_gp_app['id'], 
                    'google_play'
                )
                all_reviews += gp_revs
            
            if st.session_state.selected_ios_app:
                ios_revs = get_reviews(
                    str(st.session_state.selected_ios_app['id']), 
                    'app_store'
                )
                all_reviews += ios_revs
            
            if not all_reviews:
                st.error("Не удалось получить отзывы")
                return
            
            st.session_state.filtered_reviews = sorted(
                all_reviews,
                key=lambda x: x[0],
                reverse=True
            )
            
            with st.spinner("Анализ текста..."):
                st.session_state.analysis_data = analyze_reviews(st.session_state.filtered_reviews)
    
    # Отображение результатов
    if st.session_state.analysis_data and st.session_state.filtered_reviews:
        display_analysis(
            st.session_state.analysis_data,
            st.session_state.filtered_reviews
        )

if __name__ == "__main__":
    main()
