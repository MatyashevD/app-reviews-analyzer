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
        } for r in gp_results if r.get("score", 0) > 0]
        
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
    """Отображение результатов поиска с улучшенным интерфейсом"""
    st.subheader("🔍 Результаты поиска")
    
    if not results["google_play"] and not results["app_store"]:
        st.warning("Приложения не найдены")
        return
    
    cols = st.columns(2)
    
    with cols[0]:
        if results["google_play"]:
            st.markdown("### Google Play")
            for i, app in enumerate(results["google_play"], 1):
                container = st.container()
                container.markdown(f"""
                    **{i}. {app['title']}**  
                    Разработчик: {app['developer']}  
                    Рейтинг: {app['score']:.1f} ★  
                    """)
                if container.button(f"Выбрать Google Play", key=f"gp_{app['id']}"):
                    st.session_state.selected_gp_app = app
    
    with cols[1]:
        if results["app_store"]:
            st.markdown("### App Store")
            for i, app in enumerate(results["app_store"], 1):
                container = st.container()
                container.markdown(f"""
                    **{i}. {app['title']}**  
                    Совпадение: {app['match_score']}%  
                    Разработчик: {app['developer']}  
                    Рейтинг: {app['score']:.1f} ★  
                    """)
                if container.button(f"Выбрать App Store", key=f"ios_{app['id']}"):
                    st.session_state.selected_ios_app = app

def get_reviews(app_id: str, platform: str):
    """Получение отзывов с улучшенной обработкой ошибок"""
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

def analyze_reviews(filtered_reviews: list):
    """Анализ отзывов с добавлением расчетов рейтингов"""
    analysis = {
        'sentiments': [],
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'examples': defaultdict(list),
        'total_reviews': len(filtered_reviews),
        'gp_rating': 0.0,
        'ios_rating': 0.0
    }
    
    gp_ratings = []
    ios_ratings = []
    
    for idx, (date, text, platform, rating) in enumerate(filtered_reviews):
        analysis['platform_counts'][platform] += 1
        
        if platform == 'Google Play':
            gp_ratings.append(rating)
        else:
            ios_ratings.append(rating)
        
        phrases = extract_key_phrases(text)
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text[:100] + '...')
    
    # Расчет средних рейтингов
    if gp_ratings:
        analysis['gp_rating'] = sum(gp_ratings)/len(gp_ratings)
    if ios_ratings:
        analysis['ios_rating'] = sum(ios_ratings)/len(ios_ratings)
    
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
        
        # Защита от отсутствия данных
        gp_rating = analysis.get('gp_rating', 0)
        ios_rating = analysis.get('ios_rating', 0)
        
        cols[1].metric(
            "Google Play", 
            f"{analysis['platform_counts'].get('Google Play', 0)} отзывов",
            f"★ {gp_rating:.1f}" if gp_rating > 0 else ""
        )
        cols[2].metric(
            "App Store", 
            f"{analysis['platform_counts'].get('App Store', 0)} отзывов",
            f"★ {ios_rating:.1f}" if ios_rating > 0 else ""
        )
        
        # Остальная часть анализа без изменений...

if __name__ == "__main__":
    main()
