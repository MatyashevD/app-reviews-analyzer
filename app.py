import datetime
import streamlit as st
import requests
import pandas as pd
import spacy
import json
from openai import OpenAI
from google_play_scraper import search, reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter
from fuzzywuzzy import fuzz
from itertools import groupby

def main():
    st.set_page_config(
        page_title="Анализатор приложений",
        layout="wide",
        page_icon="📱",
        menu_items={'About': "### Анализ отзывов из Google Play и App Store"}
    )

    client = OpenAI(api_key=st.secrets.get("openai_api_key"))

    if "openai_api_key" not in st.secrets or not client.api_key:
        st.error("❌ API-ключ OpenAI не найден. Проверьте настройки секретов.")
        st.stop()

    try:
        nlp = spacy.load("ru_core_news_sm")
    except:
        spacy.cli.download("ru_core_news_sm")
        nlp = spacy.load("ru_core_news_sm")

    MAX_RESULTS = 8
    DEFAULT_LANG = 'ru'
    DEFAULT_COUNTRY = 'ru'

    def search_apps(query: str):
        results = {"google_play": [], "app_store": []}
        
        try:
            gp_results = search(query, n_hits=20, lang="ru", country="ru")
            results["google_play"] = [{
                "id": r["appId"], 
                "title": r["title"], 
                "developer": r["developer"],
                "score": r["score"], 
                "platform": 'Google Play',
                "match_score": fuzz.token_set_ratio(query, r['title']),
                "icon": r["icon"]
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
            
            sorted_results = sorted(ios_data.get("results", []), key=lambda x: x['trackName'])
            grouped = groupby(sorted_results, key=lambda x: x['trackName'])
            
            processed = []
            for name, group in grouped:
                best_match = max(group, key=lambda x: fuzz.token_set_ratio(query, x['trackName']))
                processed.append({**best_match,"match_score": fuzz.token_set_ratio(query, best_match['trackName']),"icon": best_match["artworkUrl512"].replace("512x512bb", "256x256bb")})

            processed.sort(key=lambda x: x['match_score'], reverse=True)
            
            results["app_store"] = [{
                "id": r["trackId"],
                "title": r["trackName"],
                "developer": r["artistName"],
                "score": r.get("averageUserRating", 0),
                "url": r["trackViewUrl"],
                "platform": 'App Store',
                "match_score": r['match_score'],
                "icon": r["icon"]
            } for r in processed if r.get('averageUserRating', 0) > 0][:MAX_RESULTS]
            
        except Exception as e:
            st.error(f"Ошибка поиска в App Store: {str(e)}")
        
        return results

    def display_selected_apps():
        st.subheader("✅ Выбранные приложения", divider="green")
        cols = st.columns(2)
        selected_apps = [
            st.session_state.selected_gp_app,
            st.session_state.selected_ios_app
        ]
        
        for idx, app in enumerate(selected_apps):
            if app:
                platform_style = {
                    'Google Play': {'bg': '#e8f0fe', 'color': '#1967d2'},
                    'App Store': {'bg': '#fde8ef', 'color': '#ff2d55'}
                }[app['platform']]
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style="
                        border: 2px solid {platform_style['color']};
                        border-radius: 12px;
                        padding: 16px;
                        margin: 8px 0;
                        background: {platform_style['bg']};
                        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    ">
                        <div style="font-size: 17px; font-weight: 600; color: #1a1a1a;">
                            {app['title']}
                        </div>
                        <div style="font-size: 13px; color: #666; margin: 6px 0;">
                            {app['developer']}
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="color: {platform_style['color']}; font-weight: 500;">
                                ★ {app['score']:.1f}
                            </div>
                            <div style="
                                background: {platform_style['bg']};
                                color: {platform_style['color']};
                                padding: 4px 12px;
                                border-radius: 20px;
                                font-size: 12px;
                                font-weight: 500;
                            ">
                                {app['platform']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    def display_search_results(results: dict):
        st.subheader("🔍 Результаты поиска", divider="rainbow")

        # CSS стили для карточек
        custom_css = """
            <style>
                .horizontal-scroll {
                    display: flex;
                    overflow-x: auto;
                    white-space: nowrap;
                    padding: 10px 0;
                    gap: 20px;
                }
                .app-card {
                    display: inline-block;
                    width: 260px;
                    border: 1px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 12px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                    text-align: left;
                    font-family: Arial, sans-serif;
                }
                .app-card img {
                    width: 50px; 
                    height: 50px;
                    border-radius: 12px;
                    object-fit: cover;
                }
                .platform-badge {
                    padding: 4px 12px;
                    border-radius: 20px;
                    font-size: 12px;
                    display: inline-block;
                }
            </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)

        def render_platform(platform_name, platform_data, platform_key, color, bg_color):
            if platform_data:
                st.markdown(f"### {platform_name}")

                # Создаем горизонтальный контейнер с колонками для карточек
                cols = st.columns(len(platform_data))

                for idx, app in enumerate(platform_data):
                    with cols[idx]:
                        is_selected = st.session_state.get(f"selected_{platform_key}") == app['id']
                        
                        # Отображение карточки
                        st.markdown(f"""
                        <div class="app-card">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <img src="{app.get('icon', 'https://via.placeholder.com/50')}" alt="App Icon">
                                <div>
                                    <div style="font-weight: 600; font-size: 14px;">{app['title']}</div>
                                    <div style="font-size: 12px; color: #666;">{app['developer']}</div>
                                </div>
                            </div>
                            <div style="color: {color}; font-weight: 500; font-size: 14px; margin-bottom: 10px;">
                                ★ {app['score']:.1f}
                            </div>
                            <div class="platform-badge" style="background: {bg_color}; color: {color};">
                                {platform_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Отображение кнопки "Выбрать"
                        if st.button(
                            "✓ Выбрано" if is_selected else "Выбрать",
                            key=f"{platform_key}_{app['id']}",
                            use_container_width=True
                        ):
                            st.session_state[f"selected_{platform_key}"] = app['id'] if not is_selected else None
                            st.rerun()

        # Рендеринг платформ
        render_platform("📱 App Store", results["app_store"], "ios", "#ff2d55", "#fde8ef")
        render_platform("📲 Google Play", results["google_play"], "gp", "#1967d2", "#e8f0fe")

        if not results["app_store"] and not results["google_play"]:
            st.warning("😞 Приложения не найдены")


    def get_reviews(app_id: str, platform: str, start_date: datetime.date = None, end_date: datetime.date = None):
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
                    result = [r for r in result if start_date <= r['at'].date() <= end_date]
                return [(r['at'], r['content'], 'Google Play', r['score']) for r in result]
            
            elif platform == 'app_store':
                app_store_app = AppStore(
                    country=DEFAULT_COUNTRY, 
                    app_id=app_id, 
                    app_name=st.session_state.selected_ios_app['title']
                )
                app_store_app.review(how_many=1000)
                reviews = app_store_app.reviews
                if start_date and end_date:
                    reviews = [r for r in reviews if start_date <= r['date'].date() <= end_date]
                return [(r['date'], r['review'], 'App Store', r['rating']) for r in reviews]
        
        except Exception as e:
            st.error(f"Ошибка получения отзывов: {str(e)}")
            return []

    def analyze_with_ai(reviews_text: str):
        try:
            response = client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=[{
                    "role": "system",
                    "content": """Сгенерируй анализ отзывов в формате:
                    1. Основные проблемы (5-8 пунктов)
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
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content
            return "⚠️ Анализ не удался: пустой ответ от ИИ"
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
            if platform == 'Google Play': 
                gp_ratings.append(rating)
            else: 
                ios_ratings.append(rating)
            
            doc = nlp(text)
            phrases = []
            current_phrase = []
            
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop:
                    current_phrase.append(token.text)
                else:
                    if current_phrase:
                        phrases.append(' '.join(current_phrase))
                        current_phrase = []
            
            if current_phrase:
                phrases.append(' '.join(current_phrase))
            
            for phrase in phrases:
                if 2 <= len(phrase.split()) <= 3 and len(phrase) > 4:
                    analysis['key_phrases'][phrase.lower()] += 1
        
        analysis['gp_rating'] = sum(gp_ratings)/len(gp_ratings) if gp_ratings else 0
        analysis['ios_rating'] = sum(ios_ratings)/len(ios_ratings) if ios_ratings else 0
        
        if client.api_key is not None:
            reviews_text = "\n".join([r[1] for r in filtered_reviews])
            analysis['ai_analysis'] = analyze_with_ai(reviews_text)
        
        return analysis

    def display_analysis(analysis: dict, filtered_reviews: list):
        st.header("📊 Результаты анализа", divider="rainbow")
        
        tab1, tab2 = st.tabs(["Аналитика", "Все отзывы"])
        
        with tab1:
            cols = st.columns(3)
            cols[0].metric("Всего отзывов", analysis['total_reviews'])
            cols[1].metric("Google Play", analysis['platform_counts'].get('Google Play', 0), f"★ {analysis['gp_rating']:.1f}")
            cols[2].metric("App Store", analysis['platform_counts'].get('App Store', 0), f"★ {analysis['ios_rating']:.1f}")
            
            st.subheader("Ключевые темы")
            if analysis['key_phrases']:
                for phrase, count in analysis['key_phrases'].most_common(10):
                    st.write(f"- **{phrase}** ({count} упоминаний)")
            
            if analysis['ai_analysis']:
                st.markdown("---")
                st.subheader("🤖 AI Анализ")
                st.markdown(analysis['ai_analysis'])
            else:
                st.warning("AI-анализ недоступен. Проверьте API-ключ OpenAI")
        
        with tab2:
            if filtered_reviews:
                reviews_df = pd.DataFrame([{
                    'Дата': r[0].strftime('%Y-%m-%d'),
                    'Платформа': r[2],
                    'Оценка': '★' * int(r[3]),
                    'Отзыв': r[1]
                } for r in filtered_reviews])
                
                st.dataframe(reviews_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Скачать CSV", reviews_df.to_csv(index=False), "отзывы.csv", "text/csv")
            else:
                st.warning("Нет отзывов для отображения")

    # Инициализация состояния сессии
    if 'selected_gp_app' not in st.session_state:
        st.session_state.selected_gp_app = None
    if 'selected_ios_app' not in st.session_state:
        st.session_state.selected_ios_app = None

    # Интерфейс приложения
    st.title("📱 Opini.AI - анализ отзывов мобильных приложений")
    
    # Поисковая панель
    with st.container():
        search_query = st.text_input(
            "Введите название приложения:", 
            placeholder="Например: Сбербанк или TikTok",
            help="Минимум 3 символа для поиска"
        )
        
        cols = st.columns([2, 1, 1])
        if cols[0].button("🔍 Начать поиск", use_container_width=True, type="primary"):
            if len(search_query) >= 3:
                st.session_state.search_results = search_apps(search_query)
            else:
                st.warning("⚠️ Введите минимум 3 символа")
        
        if cols[1].button("🧹 Очистить выбор", use_container_width=True):
            st.session_state.selected_gp_app = None
            st.session_state.selected_ios_app = None
            st.rerun()
        
        if cols[2].button("🔄 Сбросить все", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Отображение выбранных приложений
    if st.session_state.selected_gp_app or st.session_state.selected_ios_app:
        display_selected_apps()

    # Проверка количества выбранных приложений
    selected_count = sum([
        1 if st.session_state.selected_gp_app else 0,
        1 if st.session_state.selected_ios_app else 0
    ])

    # Отображение результатов поиска только если выбрано меньше 2 приложений
    if 'search_results' in st.session_state and selected_count < 2:
        display_search_results(st.session_state.search_results)

    # Блок анализа
    if selected_count == 2:
        with st.container():
            main_cols = st.columns([3, 3, 2])
            
            with main_cols[0]:
                start_date = st.date_input(
                    "Начальная дата",
                    value=datetime.date.today()-datetime.timedelta(days=30))
            
            with main_cols[1]:
                end_date = st.date_input(
                    "Конечная дата",
                    value=datetime.date.today())
            
            with main_cols[2]:
                if st.button(
                    "🚀 Запустить анализ",
                    use_container_width=True,
                    type="primary"
                ):
                    with st.spinner("Анализ отзывов..."):
                        all_reviews = []
                        try:
                            if st.session_state.selected_gp_app:
                                all_reviews += get_reviews(
                                    st.session_state.selected_gp_app['id'], 
                                    'google_play', 
                                    start_date, 
                                    end_date)
                            if st.session_state.selected_ios_app:
                                all_reviews += get_reviews(
                                    st.session_state.selected_ios_app['id'], 
                                    'app_store', 
                                    start_date, 
                                    end_date)
                            
                            st.session_state.filtered_reviews = sorted(all_reviews, key=lambda x: x[0], reverse=True)
                            st.session_state.analysis_data = analyze_reviews(st.session_state.filtered_reviews)
                        except Exception as e:
                            st.error(f"Ошибка анализа: {str(e)}")

    # Отображение результатов анализа
    if 'analysis_data' in st.session_state:
        display_analysis(st.session_state.analysis_data, st.session_state.filtered_reviews)

if __name__ == "__main__":
    main()
