import datetime
import streamlit as st
import requests
import pandas as pd
from google_play_scraper import search, reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import defaultdict, Counter
import spacy
from fuzzywuzzy import fuzz
from itertools import groupby
from typing import Optional

# Инициализация NLP модели
def load_nlp_model():
    try:
        return spacy.load("ru_core_news_sm")
    except:
        spacy.cli.download("ru_core_news_sm")
        return spacy.load("ru_core_news_sm")

nlp = load_nlp_model()

# Настройки поиска
MAX_RESULTS = 8
DEFAULT_LANG = 'ru'
DEFAULT_COUNTRY = 'ru'

def search_apps(query: str):
    """Поиск приложений с улучшенной обработкой ошибок"""
    results = {"google_play": [], "app_store": []}
    
    try:
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
            "url": f"https://play.google.com/store/apps/details?id={r['appId']}",
            'platform': 'Google Play',
            'match_score': fuzz.token_set_ratio(query, r['title'])
        } for r in gp_results if r.get("score", 0) > 0]
        
    except Exception as e:
        st.error(f"Ошибка поиска в Google Play: {str(e)}")
    
    try:
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
            'platform': 'App Store',
            'match_score': fuzz.token_set_ratio(query, r['trackName'])
        } for r in processed if r.get('averageUserRating', 0) > 0][:MAX_RESULTS]
        
    except Exception as e:
        st.error(f"Ошибка поиска в App Store: {str(e)}")
    
    return results

def display_search_results(results: dict):
    """Обновленный UI с возможностью выбора двух приложений"""
    st.subheader("🔍 Результаты поиска", divider="rainbow")
    
    if not results["google_play"] and not results["app_store"]:
        st.warning("Приложения не найдены")
        return

    # Все результаты
    all_results = results["google_play"] + results["app_store"]
    all_results.sort(key=lambda x: (-x['match_score'], -x['score']))

    # Стили для компактных карточек
    st.markdown("""
    <style>
        .comparison-card {
            border: 2px solid transparent;
            border-radius: 10px;
            padding: 12px;
            margin: 8px 0;
            transition: all 0.2s;
            background: white;
            cursor: pointer;
            position: relative;
        }
        .comparison-card:hover {
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }
        .selected-card {
            border-color: #4CAF50 !important;
            background: #f8fff8;
        }
        .platform-tag {
            position: absolute;
            top: 8px;
            right: 8px;
            font-size: 12px;
            padding: 4px 8px;
            border-radius: 4px;
        }
        .selection-counter {
            position: absolute;
            bottom: 8px;
            right: 8px;
            font-size: 12px;
            color: #4CAF50;
        }
        .card-title {
            font-size: 14px;
            font-weight: 500;
            margin-right: 40px;
        }
        .card-developer {
            font-size: 12px;
            color: #666;
            margin-top: 4px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Отображение карточек в 3 колонки
    cols = st.columns(3)
    for idx, app in enumerate(all_results):
        with cols[idx % 3]:
            is_selected_gp = app['platform'] == 'Google Play' and \
                st.session_state.selected_gp_app and \
                st.session_state.selected_gp_app['id'] == app['id']
            
            is_selected_ios = app['platform'] == 'App Store' and \
                st.session_state.selected_ios_app and \
                st.session_state.selected_ios_app['id'] == app['id']
            
            is_selected = is_selected_gp or is_selected_ios
            platform_color = "#4285f4" if app['platform'] == 'Google Play' else "#000000"
            selection_count = (1 if st.session_state.selected_gp_app else 0) + \
                             (1 if st.session_state.selected_ios_app else 0)

            card_html = f"""
            <div class="comparison-card {'selected-card' if is_selected else ''}">
                <div class="platform-tag" style="background:{platform_color}10;color:{platform_color}">
                    {app['platform']}
                </div>
                <div class="card-title">{app['title']}</div>
                <div class="card-developer">{app['developer']}</div>
                <div style="margin-top:8px;">
                    <span style="color:#ff9800;">★ {app['score']:.1f}</span>
                    <span style="float:right;font-size:12px;color:#666">{app['match_score']}%</span>
                </div>
                {f'<div class="selection-counter">Выбрано ({selection_count}/2)</div>' if is_selected else ''}
            </div>
            """
            
            st.markdown(card_html, unsafe_allow_html=True)
            
            # Обработка выбора
            if st.button(
                "✓" if is_selected else "Выбрать",
                key=f"select_{app['id']}",
                type="primary" if is_selected else "secondary",
                use_container_width=True
            ):
                if app['platform'] == 'Google Play':
                    if st.session_state.selected_gp_app and st.session_state.selected_gp_app['id'] == app['id']:
                        st.session_state.selected_gp_app = None
                    else:
                        st.session_state.selected_gp_app = app
                else:
                    if st.session_state.selected_ios_app and st.session_state.selected_ios_app['id'] == app['id']:
                        st.session_state.selected_ios_app = None
                    else:
                        st.session_state.selected_ios_app = app
                st.rerun()

    # Панель выбранных приложений
    selected_apps = []
    if st.session_state.selected_gp_app:
        selected_apps.append(st.session_state.selected_gp_app)
    if st.session_state.selected_ios_app:
        selected_apps.append(st.session_state.selected_ios_app)

    if selected_apps:
        st.markdown("---")
        with st.container():
            cols = st.columns([4,1])
            with cols[0]:
                st.subheader("✅ Выбранные приложения")
                for app in selected_apps:
                    platform_icon = "📱" if app['platform'] == 'Google Play' else ""
                    st.markdown(f"""
                    {platform_icon} **{app['title']}**  
                    *Рейтинг:* ★ {app['score']:.1f} | *Совпадение:* {app['match_score']}%  
                    *Разработчик:* {app['developer']}
                    """)
            
            with cols[1]:
                st.write("")  # Вертикальное выравнивание
                st.write("")
                if st.button("Очистить выбор", use_container_width=True):
                    st.session_state.selected_gp_app = None
                    st.session_state.selected_ios_app = None
                    st.rerun()

    # Валидация выбора
    if (st.session_state.selected_gp_app or st.session_state.selected_ios_app) and \
        not (st.session_state.selected_gp_app and st.session_state.selected_ios_app):
        st.warning("⚠️ Для сравнения необходимо выбрать по одному приложению из каждого магазина")

def get_reviews(app_id: str, platform: str, 
                start_date: Optional[datetime.date] = None, 
                end_date: Optional[datetime.date] = None):
    try:
        if platform == 'google_play':
            result, _ = gp_reviews(
                app_id,
                lang=DEFAULT_LANG,
                country=DEFAULT_COUNTRY,
                count=1000,
                sort=Sort.NEWEST
            )
            if start_date and end_date:
                result = [
                    r for r in result 
                    if start_date <= r['at'].date() <= end_date
                ]
            return [(r['at'], r['content'], 'Google Play', r['score']) for r in result]
        else:
            app_store_app = AppStore(
                country=DEFAULT_COUNTRY, 
                app_id=app_id, 
                app_name=st.session_state.selected_ios_app['title']
            )
            app_store_app.review(how_many=1000)
            reviews = app_store_app.reviews
            if start_date and end_date:
                reviews = [
                    r for r in reviews 
                    if start_date <= r['date'].date() <= end_date
                ]
            return [(r['date'], r['review'], 'App Store', r['rating']) for r in reviews]
    except Exception as e:
        st.error(f"Ошибка получения отзывов: {str(e)}")
        return []

def analyze_reviews(filtered_reviews: list):
    analysis = {
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
        
        # Заменяем использование noun_chunks на ручное извлечение фраз
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
        filtered_phrases = [
            phrase.strip().lower()
            for phrase in phrases
            if 2 <= len(phrase.split()) <= 3
            and len(phrase) > 4
        ]
        
        for phrase in filtered_phrases:
            analysis['key_phrases'][phrase] += 1
            if len(analysis['examples'][phrase]) < 3:
                analysis['examples'][phrase].append(text[:100] + '...')
    
    analysis['gp_rating'] = sum(gp_ratings)/len(gp_ratings) if gp_ratings else 0
    analysis['ios_rating'] = sum(ios_ratings)/len(ios_ratings) if ios_ratings else 0
    
    return analysis


def display_analysis(analysis: dict, filtered_reviews: list):
    st.header("📊 Результаты анализа", divider="rainbow")
    
    tab1, tab2 = st.tabs(["📈 Аналитика", "📋 Все отзывы"])
    
    with tab1:
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", analysis['total_reviews'])
        
        gp_rating = analysis.get('gp_rating', 0)
        ios_rating = analysis.get('ios_rating', 0)
        
        cols[1].metric(
            "Google Play", 
            f"{analysis['platform_counts'].get('Google Play', 0)}",
            f"★ {gp_rating:.1f}" if gp_rating > 0 else ""
        )
        cols[2].metric(
            "App Store", 
            f"{analysis['platform_counts'].get('App Store', 0)}",
            f"★ {ios_rating:.1f}" if ios_rating > 0 else ""
        )
        
        st.subheader("📊 Распределение оценок")
        try:
            ratings = [r[3] for r in filtered_reviews]
            rating_data = pd.DataFrame({
                'Оценка': ratings,
                'Платформа': [r[2] for r in filtered_reviews]
            })
            st.bar_chart(rating_data, x='Оценка', y='Платформа', color='Платформа')
        except Exception as e:
            st.warning("Не удалось построить график оценок")
        
        st.subheader("🔍 Ключевые темы")
        if analysis['key_phrases']:
            top_phrases = analysis['key_phrases'].most_common(15)
            
            for phrase, count in top_phrases:
                with st.expander(f"{phrase.capitalize()} ({count} упоминаний)"):
                    examples = analysis['examples'].get(phrase, [])
                    if examples:
                        st.caption("Примеры отзывов:")
                        for ex in examples[:3]:
                            st.markdown(f"- {ex}")
                    else:
                        st.caption("Нет примеров")
        else:
            st.info("Ключевые темы не обнаружены")
    
    with tab2:
        st.subheader("📄 Все отзывы")
        reviews_df = pd.DataFrame([{
            'Дата': r[0].strftime('%Y-%m-%d') if isinstance(r[0], datetime.datetime) else r[0],
            'Платформа': r[2],
            'Оценка': '★' * int(r[3]),
            'Отзыв': r[1]
        } for r in filtered_reviews])
        
        st.dataframe(
            reviews_df,
            column_config={
                "Оценка": st.column_config.TextColumn(width="small"),
                "Отзыв": st.column_config.TextColumn(width="large")
            },
            height=600,
            use_container_width=True,
            hide_index=True
        )
        
        csv = reviews_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Скачать CSV",
            data=csv,
            file_name='отзывы.csv',
            mime='text/csv',
            use_container_width=True
        )

def main():
    st.set_page_config(
        page_title="Анализатор приложений",
        layout="wide",
        page_icon="📱",
        menu_items={'About': "### Анализатор мобильных приложений v2.0"}
    )
    
    st.title("📱 Анализатор мобильных приложений")
    st.caption("Сравнивайте отзывы из Google Play и App Store в одном интерфейсе")
    
    # Инициализация состояния
    if 'selected_gp_app' not in st.session_state:
        st.session_state.selected_gp_app = None
    if 'selected_ios_app' not in st.session_state:
        st.session_state.selected_ios_app = None
    
    # Поисковая панель
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            search_query = st.text_input(
                "Поиск приложений:",
                placeholder="Введите название приложения...",
                help="Начните вводить название приложения (минимум 3 символа)"
            )
        with cols[1]:
            st.write("")
            st.write("")
            if st.button("🔍 Найти", use_container_width=True):
                if len(search_query) >= 3:
                    with st.spinner("Поиск..."):
                        st.session_state.search_results = search_apps(search_query)
                else:
                    st.warning("Введите минимум 3 символа")
    
    # Отображение результатов поиска
    if 'search_results' in st.session_state and st.session_state.search_results:
        display_search_results(st.session_state.search_results)
    
    # Выбор периода и запуск анализа
    if st.session_state.selected_gp_app or st.session_state.selected_ios_app:
        st.divider()
        
        with st.container():
            st.subheader("🛠 Настройки анализа")
            cols = st.columns([2, 2, 3])
            
            with cols[0]:
                start_date = st.date_input(
                    "Дата начала",
                    value=datetime.date.today() - datetime.timedelta(days=30),
                    key="start_date"
                )
            with cols[1]:
                end_date = st.date_input(
                    "Дата окончания",
                    value=datetime.date.today(),
                    key="end_date"
                )
            with cols[2]:
                st.write("")
                if st.button("🚀 Запустить анализ", use_container_width=True):
                    if start_date > end_date:
                        st.error("Дата начала должна быть раньше даты окончания")
                    else:
                        with st.spinner("Сбор данных..."):
                            all_reviews = []
                            
                            if st.session_state.selected_gp_app:
                                gp_revs = get_reviews(
                                    st.session_state.selected_gp_app['id'], 
                                    'google_play',
                                    start_date,
                                    end_date
                                )
                                all_reviews += gp_revs
                            
                            if st.session_state.selected_ios_app:
                                ios_revs = get_reviews(
                                    str(st.session_state.selected_ios_app['id']), 
                                    'app_store',
                                    start_date,
                                    end_date
                                )
                                all_reviews += ios_revs
                            
                            if all_reviews:
                                st.session_state.filtered_reviews = sorted(
                                    all_reviews,
                                    key=lambda x: x[0],
                                    reverse=True
                                )
                                st.session_state.analysis_data = analyze_reviews(
                                    st.session_state.filtered_reviews
                                )
                            else:
                                st.error("Не найдено отзывов за выбранный период")
    
    # Отображение результатов анализа
    if 'analysis_data' in st.session_state and st.session_state.analysis_data:
        st.divider()
        display_analysis(st.session_state.analysis_data, st.session_state.filtered_reviews)
        
        if st.button("🔄 Новый анализ", use_container_width=True):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
