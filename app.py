import datetime
import os
import streamlit as st
import requests
import pandas as pd
import openai
from google_play_scraper import search, reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import defaultdict, Counter
import spacy
from fuzzywuzzy import fuzz
from itertools import groupby
from typing import Optional
from dotenv import load_dotenv

# Загрузка API ключа
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_nlp_model():
    try:
        return spacy.load("ru_core_news_sm")
    except:
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

nlp = load_nlp_model()

MAX_RESULTS = 8
DEFAULT_LANG = 'ru'
DEFAULT_COUNTRY = 'ru'

def search_apps(query: str):
    results = {"google_play": [], "app_store": []}
    
    try:
        gp_results = search(query, lang=DEFAULT_LANG, country=DEFAULT_COUNTRY, n_hits=MAX_RESULTS)
        results["google_play"] = [{
            "id": r["appId"], "title": r["title"], "developer": r["developer"],
            "score": r["score"], "platform": 'Google Play',
            "match_score": fuzz.token_set_ratio(query, r['title'])
        } for r in gp_results if r.get("score", 0) > 0]
        
    except Exception as e:
        st.error(f"Ошибка поиска в Google Play: {str(e)}")
    
    try:
        itunes_response = requests.get(
            "https://itunes.apple.com/search",
            params={"term": query, "country": DEFAULT_COUNTRY, "media": "software", "limit": 20},
            headers={"User-Agent": "Mozilla/5.0"}
        )
        ios_data = itunes_response.json()
        
        sorted_results = sorted(ios_data.get("results", []), key=lambda x: x['trackName'])
        processed = [max(group, key=lambda x: fuzz.token_set_ratio(query, x['trackName'])) 
                   for _, group in groupby(sorted_results, key=lambda x: x['trackName'])]
        
        results["app_store"] = [{
            "id": r["trackId"], "title": r["trackName"], 
            "developer": r["artistName"], "score": r.get("averageUserRating", 0),
            "platform": 'App Store', "match_score": fuzz.token_set_ratio(query, r['trackName'])
        } for r in sorted(processed, key=lambda x: x['match_score'], reverse=True)[:MAX_RESULTS]]
        
    except Exception as e:
        st.error(f"Ошибка поиска в App Store: {str(e)}")
    
    return results

def display_search_results(results: dict):
    st.subheader("🔍 Результаты поиска", divider="rainbow")
    
    st.markdown("""
    <style>
        .mobile-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 8px;
            margin: 6px 0;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
        }
        .selected-card { border: 2px solid #4CAF50; background: #f8fff8; }
        .app-title { font-size: 14px; font-weight: 600; color: #1a1a1a; }
        .meta-info { display: flex; justify-content: space-between; margin-top: 8px; }
        .rating { color: #ff9800; font-weight: 500; }
        .platform-tag { background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }
    </style>
    """, unsafe_allow_html=True)

    if not results["google_play"] and not results["app_store"]:
        st.warning("Приложения не найдены")
        return

    all_results = results["google_play"] + results["app_store"]
    all_results.sort(key=lambda x: (-x['match_score'], -x['score']))

    for app in all_results:
        is_selected = any([
            st.session_state.selected_gp_app and app['id'] == st.session_state.selected_gp_app['id'],
            st.session_state.selected_ios_app and app['id'] == st.session_state.selected_ios_app['id']
        ])
        
        card_html = f"""
        <div class="mobile-card {'selected-card' if is_selected else ''}">
            <div class="app-title">{app['title']}</div>
            <div class="meta-info">
                <div>
                    <span class="rating">★ {app['score']:.1f}</span>
                    <span style="color:#666;font-size:12px">{app['match_score']}%</span>
                </div>
                <div class="platform-tag">{app['platform']}</div>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        if st.button(
            "✓" if is_selected else " ",
            key=f"select_{app['id']}",
            type="primary" if is_selected else "secondary",
            use_container_width=True
        ):
            if app['platform'] == 'Google Play':
                st.session_state.selected_gp_app = app if not is_selected else None
            else:
                st.session_state.selected_ios_app = app if not is_selected else None
            st.rerun()

def analyze_with_ai(reviews_text: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": """Сгенерируй анализ отзывов в формате:
                1. Основные проблемы (3-5 пунктов)
                2. Распределение тональности (проценты)
                3. Рекомендации по улучшению
                
                Используй маркдаун для оформления"""
            }, {
                "role": "user",
                "content": f"Отзывы:\n{reviews_text[:10000]}"
            }],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"AI анализ недоступен: {str(e)}")
        return None

def analyze_reviews(filtered_reviews: list):
    analysis = {
        'key_phrases': Counter(),
        'platform_counts': Counter(),
        'total_reviews': len(filtered_reviews),
        'gp_rating': 0.0,
        'ios_rating': 0.0,
        'ai_analysis': None
    }
    
    gp_ratings, ios_ratings = [], []
    
    for date, text, platform, rating in filtered_reviews:
        analysis['platform_counts'][platform] += 1
        if platform == 'Google Play': gp_ratings.append(rating)
        else: ios_ratings.append(rating)
        
        doc = nlp(text)
        phrases = [
            chunk.text.lower() 
            for chunk in doc.noun_chunks 
            if 2 <= len(chunk.text.split()) <= 3
        ]
        for phrase in phrases:
            analysis['key_phrases'][phrase] += 1
    
    analysis['gp_rating'] = sum(gp_ratings)/len(gp_ratings) if gp_ratings else 0
    analysis['ios_rating'] = sum(ios_ratings)/len(ios_ratings) if ios_ratings else 0
    
    if openai.api_key:
        reviews_text = "\n".join([r[1] for r in filtered_reviews])
        analysis['ai_analysis'] = analyze_with_ai(reviews_text)
    
    return analysis

def display_analysis(analysis: dict, filtered_reviews: list):
    st.header("📊 Результаты анализа", divider="rainbow")
    
    tab1, tab2 = st.tabs(["Аналитика", "Все отзывы"])
    
    with tab1:
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", analysis['total_reviews'])
        cols[1].metric("Google Play", analysis['platform_counts']['Google Play'], f"★ {analysis['gp_rating']:.1f}")
        cols[2].metric("App Store", analysis['platform_counts']['App Store'], f"★ {analysis['ios_rating']:.1f}")
        
        st.subheader("Ключевые темы")
        if analysis['key_phrases']:
            for phrase, count in analysis['key_phrases'].most_common(10):
                st.write(f"- **{phrase}** ({count} упоминаний)")
        
        if analysis['ai_analysis']:
            st.markdown("---")
            st.subheader("🤖 ИИ Анализ")
            st.markdown(analysis['ai_analysis'])
    
    with tab2:
        reviews_df = pd.DataFrame([{
            'Дата': r[0].strftime('%Y-%m-%d'),
            'Платформа': r[2],
            'Оценка': '★' * int(r[3]),
            'Отзыв': r[1]
        } for r in filtered_reviews])
        
        st.dataframe(reviews_df, use_container_width=True, hide_index=True)
        st.download_button("📥 Скачать", reviews_df.to_csv(index=False), "отзывы.csv", "text/csv")

def main():
    st.set_page_config(page_title="Анализатор приложений", layout="wide", page_icon="📱")
    st.title("📱 Анализатор мобильных приложений")
    
    if 'selected_gp_app' not in st.session_state:
        st.session_state.selected_gp_app = None
    if 'selected_ios_app' not in st.session_state:
        st.session_state.selected_ios_app = None
    
    with st.container():
        cols = st.columns([4, 1])
        search_query = cols[0].text_input("Поиск приложений:", placeholder="Введите название...")
        if cols[1].button("🔍 Найти", use_container_width=True) and len(search_query) >= 3:
            st.session_state.search_results = search_apps(search_query)
    
    if 'search_results' in st.session_state:
        display_search_results(st.session_state.search_results)
    
    if st.session_state.selected_gp_app and st.session_state.selected_ios_app:
        with st.container():
            cols = st.columns(3)
            start_date = cols[0].date_input("Начальная дата", datetime.date.today() - datetime.timedelta(days=30))
            end_date = cols[1].date_input("Конечная дата", datetime.date.today())
            if cols[2].button("🚀 Анализировать", use_container_width=True):
                with st.spinner("Анализ..."):
                    all_reviews = []
                    try:
                        if st.session_state.selected_gp_app:
                            all_reviews += get_reviews(st.session_state.selected_gp_app['id'], 'google_play', start_date, end_date)
                        if st.session_state.selected_ios_app:
                            all_reviews += get_reviews(st.session_state.selected_ios_app['id'], 'app_store', start_date, end_date)
                        
                        st.session_state.filtered_reviews = sorted(all_reviews, key=lambda x: x[0], reverse=True)
                        st.session_state.analysis_data = analyze_reviews(st.session_state.filtered_reviews)
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
    
    if 'analysis_data' in st.session_state:
        display_analysis(st.session_state.analysis_data, st.session_state.filtered_reviews)

if __name__ == "__main__":
    main()
