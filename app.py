import datetime
import re
import streamlit as st
import requests
import pandas as pd
from fuzzywuzzy import fuzz
from itertools import groupby
from bs4 import BeautifulSoup
from google_play_scraper import search, app, reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import defaultdict, Counter
import spacy

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
        # Поиск в App Store через iTunes API
        itunes_response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": query,
                "country": "RU",
                "media": "software",
                "limit": 20,
                "entity": "software,iPadSoftware",
                "lang": "ru_ru"
            },
            headers={"User-Agent": "Mozilla/5.0"}
        )
        ios_data = itunes_response.json()
        
        # Группировка по названию приложения
        sorted_results = sorted(ios_data.get("results", []), 
                              key=lambda x: x['trackName'])
        grouped = groupby(sorted_results, key=lambda x: x['trackName'])
        
        # Фильтрация и выбор лучшего совпадения
        processed = []
        for name, group in grouped:
            best_match = max(group, 
                           key=lambda x: fuzz.token_set_ratio(query, x['trackName']))
            processed.append(best_match)

        # Сортировка по релевантности
        processed.sort(key=lambda x: fuzz.token_set_ratio(query, x['trackName']), 
                      reverse=True)
        
        results["app_store"] = [{
            "id": r["trackId"],
            "title": r["trackName"],
            "developer": r["artistName"],
            "score": r.get("averageUserRating", 0),
            "url": r["trackViewUrl"],
            "match_score": fuzz.token_set_ratio(query, r['trackName'])
        } for r in processed if r['match_score'] > 65][:5]
        
    except Exception as e:
        st.error(f"Ошибка поиска в App Store: {str(e)}")
    
    return results

def display_search_results(results: dict):
    """Обновленное отображение результатов"""
    if results["app_store"]:
        st.markdown("### App Store (лучшие совпадения)")
        for i, app in enumerate(results["app_store"], 1):
            with st.expander(f"{i}. {app['title']} ({app['match_score']}% совпадение)"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                st.write(f"**Ссылка:** {app['url']}")
                if st.button(f"Выбрать", key=f"ios_{app['id']}"):
                    st.session_state.selected_ios_app = app

    
    # App Store
    if results["app_store"]:
        st.markdown("### App Store")
        for i, app in enumerate(results["app_store"], 1):
            with st.expander(f"{i}. {app['title']}"):
                st.write(f"**Разработчик:** {app['developer']}")
                st.write(f"**Рейтинг:** {app['score']:.1f} ★")
                st.write(f"**Ссылка:** {app['url']}")
                if st.button(f"Выбрать", key=f"ios_{app['id']}"):
                    st.session_state.selected_ios_app = app

def get_reviews(app_id: str, platform: str):
    """Получение отзывов для выбранного приложения"""
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
                app_name=st.session_state[f"selected_{platform}_app"]['title']
            )
            app_store_app.review(how_many=100)
            return [(r['date'], r['review'], 'App Store', r['rating']) for r in app_store_app.reviews]
    except Exception as e:
        st.error(f"Ошибка получения отзывов: {str(e)}")
        return []

def analyze_reviews(filtered_reviews: list):
    """Анализ отзывов с использованием NLP"""
    analysis = {
        'sentiments': [],
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'examples': defaultdict(list)
    }
    
    for idx, (date, text, platform, rating) in enumerate(filtered_reviews):
        analysis['platform_counts'][platform] += 1
        
        # Извлечение ключевых фраз
        doc = nlp(text)
        phrases = [chunk.text for chunk in doc.noun_chunks if 2 <= len(chunk.text.split()) <= 4]
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text[:150] + '...')
    
    return analysis

def display_analysis(analysis: dict, filtered_reviews: list):
    """Отображение результатов анализа"""
    st.header("📊 Результаты анализа")
    
    # Основные метрики
    cols = st.columns(2)
    cols[0].metric("Всего отзывов", len(filtered_reviews))
    cols[1].metric("Платформы", 
                  f"Google Play: {analysis['platform_counts']['Google Play']} | App Store: {analysis['platform_counts']['App Store']}")
    
    # Ключевые фразы
    st.subheader("🔑 Ключевые темы")
    if analysis['key_phrases']:
        phrases_df = pd.DataFrame(
            analysis['key_phrases'].most_common(15),
            columns=['Фраза', 'Количество']
        )
        st.dataframe(
            phrases_df.style.background_gradient(subset=['Количество'], cmap='Blues'),
            height=400
        )
    else:
        st.info("Ключевые темы не обнаружены")
    
    # Примеры отзывов
    st.subheader("📋 Последние отзывы")
    reviews_df = pd.DataFrame([{
        'Дата': r[0].strftime('%Y-%m-%d'),
        'Платформа': r[2],
        'Оценка': '★' * int(r[3]),
        'Текст': r[1]
    } for r in filtered_reviews[:10]])
    st.dataframe(reviews_df, height=500)

def main():
    st.set_page_config(
        page_title="Анализатор приложений",
        layout="wide",
        menu_items={'About': "### Анализатор мобильных приложений v3.0"}
    )
    st.title("📱 Анализатор мобильных приложений")
    
    # Поисковая строка
    search_query = st.text_input(
        "Введите название приложения:",
        placeholder="Например: ВКонтакте, СберБанк",
        key="search_input"
    )
    
    # Инициализация состояния
    session_defaults = {
        'search_results': None,
        'selected_gp_app': None,
        'selected_ios_app': None,
        'analysis_data': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
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
    
    # Отображение выбранных приложений
    selected_apps = []
    if st.session_state.selected_gp_app:
        selected_apps.append(f"Google Play: {st.session_state.selected_gp_app['title']}")
    if st.session_state.selected_ios_app:
        selected_apps.append(f"App Store: {st.session_state.selected_ios_app['title']}")
    
    if selected_apps:
        st.success("✅ Выбрано: " + " | ".join(selected_apps))
    
    # Кнопка анализа
    if selected_apps and st.button("🚀 Начать анализ отзывов", type="primary"):
        all_reviews = []
        
        with st.spinner("Сбор отзывов..."):
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
            
            # Фильтрация по дате
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            end_date = datetime.datetime.now()
            filtered_reviews = [
                r for r in all_reviews 
                if start_date <= r[0] <= end_date
            ]
            
            with st.spinner("Анализ текста..."):
                analysis = analyze_reviews(filtered_reviews)
                st.session_state.analysis_data = analysis
        
        if st.session_state.analysis_data:
            display_analysis(
                st.session_state.analysis_data, 
                filtered_reviews
            )

if __name__ == "__main__":
    main()
