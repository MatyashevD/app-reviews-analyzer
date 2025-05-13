import time
import json
import random
import datetime
import streamlit as st
import requests
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from app_store_web_scraper import AppStoreEntry, AppStoreSession
from openai import OpenAI
from google_play_scraper import search, reviews as gp_reviews, Sort
from collections import Counter
from rapidfuzz import fuzz
from itertools import groupby
from urllib.parse import urlparse

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

    MAX_RESULTS = 5
    DEFAULT_LANG = 'ru'
    DEFAULT_COUNTRY = 'ru'
    GOOGLE_PLAY_MAX_REVIEWS = 10000
    APP_STORE_MAX_REVIEWS = 500

    def extract_app_store_id(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        parts = path.split('/')
        for part in reversed(parts):
            if part.startswith('id') and part[2:].isdigit():
                return part[2:]
            if part.isdigit():
                return part
        return None

    def search_apps(query: str):
        results = {"google_play": [], "app_store": []}
        normalized_query = query.strip().lower()
        
        # Поиск в Google Play
        try:
            gp_results = search(normalized_query, n_hits=20, lang="ru", country="ru")
            results["google_play"] = [{
                "id": r["appId"], 
                "title": r["title"], 
                "developer": r["developer"],
                "score": r["score"],
                "release_date": datetime.datetime.strptime(r['released'], "%b %d, %Y").date() if r.get('released') else None,
                "platform": 'Google Play',
                "match_score": fuzz.token_set_ratio(normalized_query, r['title'].lower()),
                "icon": r["icon"]
            } for r in gp_results if r.get("score", 0) > 0]
                    
        except Exception as e:
            st.error(f"Ошибка поиска в Google Play: {str(e)}")
        
        # Поиск в App Store
        try:
            itunes_response = requests.get(
                "https://itunes.apple.com/search",
                params={
                    "term": normalized_query,
                    "country": DEFAULT_COUNTRY,
                    "media": "software",
                    "limit": 50,
                    "entity": "software,iPadSoftware",
                    "lang": "ru_ru"
                },
                headers={"User-Agent": "Mozilla/5.0"}
            )
            ios_data = itunes_response.json()
            
            processed = []
            for r in ios_data.get("results", []):
                try:
                    release_date = (
                        datetime.datetime.strptime(
                            r['currentVersionReleaseDate'].replace('Z', '+00:00'), 
                            '%Y-%m-%dT%H:%M:%S%z'
                        ).date()
                    ) if r.get('currentVersionReleaseDate') else None
                    
                    processed.append({
                        "id": str(r["trackId"]),
                        "app_store_id": extract_app_store_id(r["trackViewUrl"]),
                        "title": r["trackName"],
                        "developer": r["artistName"],
                        "score": r.get("averageUserRating", 0),
                        "release_date": release_date,
                        "url": r["trackViewUrl"],
                        "platform": 'App Store',
                        "match_score": fuzz.token_set_ratio(
                            normalized_query,
                            r['trackName'].strip().lower()
                        ),
                        "icon": r["artworkUrl512"].replace("512x512bb", "256x256bb")
                    })
                except Exception as e:
                    continue

            results["app_store"] = sorted(
                [r for r in processed if r['score'] > 0],
                key=lambda x: (-x['match_score'], -x['score']),
            )[:MAX_RESULTS]
            
        except Exception as e:
            st.error(f"Ошибка поиска в App Store: {str(e)}")
        
        return results

    def display_selected_apps():
        st.subheader("✅ Выбранные приложения", divider="green")
        cols = st.columns(2)
        selected_apps = [
            st.session_state.get('selected_gp_app'),
            st.session_state.get('selected_ios_app')
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

        custom_css = """
            <style>
                .horizontal-scroll {
                    display: flex;
                    overflow-x: auto;
                    padding: 10px 0;
                    gap: 20px;
                }
                .app-card {
                    width: 400px;
                    border: 1px solid #e0e0e0;
                    border-radius: 12px;
                    padding: 12px;
                    background: white;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                }
                .app-card img {
                    width: 50px; 
                    height: 50px;
                    border-radius: 12px;
                }
            </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        def render_platform(platform_name, platform_data, platform_key, color, bg_color):
            if platform_data:
                st.markdown(f"### {platform_name}")
                cols = st.columns(len(platform_data))
                for idx, app in enumerate(platform_data):
                    with cols[idx]:
                        # Исправленная строка:
                        selected_app = st.session_state.get(f"selected_{platform_key}_app") or {}
                        is_selected = selected_app.get('id') == app['id']
                        
                        st.markdown(f"""
                        <div class="app-card">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <img src="{app.get('icon', 'https://via.placeholder.com/50')}">
                                <div>
                                    <div style="font-weight: 600; font-size: 14px; color: #2e2e2e;">{app['title']}</div>
                                    <div style="font-size: 12px; color: #a8a8a8;">{app['developer']}</div>
                                </div>
                            </div>
                            <div style="color: {color}; font-weight: 500; margin-bottom: 10px;">
                                ★ {app['score']:.1f}
                            </div>
                            <div style="background: {bg_color}; color: {color}; 
                                padding: 4px 12px; border-radius: 20px; font-size: 12px;">
                                {platform_name}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        if st.button(
                            "✓ Выбрано" if is_selected else "Выбрать",
                            key=f"{platform_key}_{app['id']}",
                            use_container_width=True
                        ):
                            if platform_key == "gp":
                                new_selection = app if not is_selected else None
                                st.session_state.selected_gp_app = new_selection
                                if new_selection and new_selection.get('release_date'):
                                    st.session_state.gp_release_dates = [{
                                        'date': new_selection['release_date'],
                                        'platform': 'Google Play'
                                    }]
                            elif platform_key == "ios":
                                new_selection = app if not is_selected else None
                                st.session_state.selected_ios_app = new_selection
                                if new_selection and new_selection.get('release_date'):
                                    st.session_state.ios_release_dates = [{
                                        'date': new_selection['release_date'],
                                        'platform': 'App Store'
                                    }]
                            st.rerun()

        render_platform(" App Store", results["app_store"], "ios", "#399eff", "#cce2ff")
        render_platform("📲 Google Play", results["google_play"], "gp", "#36c55f", "#e3ffeb")

        if not results["app_store"] and not results["google_play"]:
            st.warning("😞 Приложения не найдены")

    def get_reviews(app_id: str, platform: str, start_date: datetime.date, end_date: datetime.date):
        try:
            if platform == 'google_play':
                # Запрашиваем отзывы порциями по 100 штук
                batch_size = 100
                all_reviews = []
                continuation_token = None
                date_filter_enabled = False
    
                while True:
                    result, continuation_token = gp_reviews(
                        app_id,
                        lang=DEFAULT_LANG,
                        country=DEFAULT_COUNTRY,
                        count=batch_size,
                        sort=Sort.NEWEST,
                        continuation_token=continuation_token
                    )
    
                    # Фильтрация на лету с прерыванием при выходе за диапазон
                    for r in result:
                        review_date = r['at'].date()
                        if review_date < start_date:
                            date_filter_enabled = True
                            break
                        if start_date <= review_date <= end_date:
                            all_reviews.append((
                                r['at'].replace(tzinfo=None),
                                r['content'],
                                'Google Play',
                                r['score']
                            ))
    
                    if date_filter_enabled or not continuation_token or len(all_reviews) >= 1000:
                        break
    
                    time.sleep(1)  # Защита от блокировки
    
                return all_reviews
    
            elif platform == 'app_store':
                # Оптимизированный запрос для App Store
                selected_app = st.session_state.get('selected_ios_app')
                if not selected_app or not selected_app.get('app_store_id'):
                    st.error("Не выбрано приложение из App Store")
                    return []                

                session = AppStoreSession(
                    delay=0.5,
                    retries=3
                )
    
                app_entry = AppStoreEntry(
                    app_id=selected_app['app_store_id'],
                    country=DEFAULT_COUNTRY.lower(),
                    session=session
                )
    
                # Собираем только свежие отзывы
                reviews = []
                for review in app_entry.reviews(limit=500):
                    review_date = review.date.date()
                    
                    if start_date <= review_date <= end_date
                        reviews.append((
                            review.date.replace(tzinfo=None),
                            review.review,
                            'App Store',
                            review.rating
                        ))
                return reviews
    
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
            return response.choices[0].message.content if response.choices else "⚠️ Анализ не удался"
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
        
        for _, text, platform, rating in filtered_reviews:
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
        
        if client.api_key:
            reviews_text = "\n".join([r[1] for r in filtered_reviews[:2000]])
            analysis['ai_analysis'] = analyze_with_ai(reviews_text)
        
        return analysis

    def display_analysis(analysis: dict, filtered_reviews: list, start_date: datetime.date, end_date: datetime.date):
        st.header("📊 Результаты анализа", divider="rainbow")
        
        tab1, tab2, tab3 = st.tabs(["Аналитика", "Все отзывы", "Графики"])
        
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
        
        with tab3:
            selected_platform = st.radio(
                "Выберите платформу:",
                ["Google Play", "App Store"],
                horizontal=True
            )
            
            platform_filtered = [
                (r[0].date(), r[3]) 
                for r in filtered_reviews 
                if r[2] == selected_platform
            ]
            
            release_dates = []
            if selected_platform == "Google Play":
                release_dates = st.session_state.get('gp_release_dates', [])
            else:
                release_dates = st.session_state.get('ios_release_dates', [])
            
            if not platform_filtered:
                st.warning(f"Нет отзывов для {selected_platform}")
                return
            
            df = pd.DataFrame(platform_filtered, columns=['date', 'rating'])
            daily_ratings = df.groupby('date')['rating'].value_counts().unstack().fillna(0)
            
            colors = {
                1: '#FF0000', 2: '#FFA500', 3: '#FFFF00', 
                4: '#90EE90', 5: '#008000'
            }
            platform_color = '#36c55f' if selected_platform == "Google Play" else '#399eff'
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bottom = None
            
            for rating in [1, 2, 3, 4, 5]:
                if rating in daily_ratings.columns:
                    ax.bar(
                        daily_ratings.index,
                        daily_ratings[rating],
                        color=colors[rating],
                        label=f'{rating} звезд',
                        bottom=bottom
                    )
                    bottom = daily_ratings[rating] if bottom is None else bottom + daily_ratings[rating]
            
            if release_dates:
                max_y = daily_ratings.sum(axis=1).max()
                for item in release_dates:
                    try:
                        date = item['date']
                        if start_date <= date <= end_date:
                            ax.scatter(
                                mdates.date2num(date),
                                max_y * 1.1,
                                color=platform_color,
                                marker='*',
                                s=200,
                                zorder=3,
                                label='Дата релиза'
                            )
                    except Exception as e:
                        st.error(f"Ошибка в дате релиза: {str(e)}")
            
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(), title='Легенда', bbox_to_anchor=(1.05, 1))
            plt.title(f'Оценки и релизы ({selected_platform})')
            plt.tight_layout()
            st.pyplot(fig)

    if 'selected_gp_app' not in st.session_state:
        st.session_state.selected_gp_app = None
    if 'selected_ios_app' not in st.session_state:
        st.session_state.selected_ios_app = None

    st.title("📱 Opini.AI - анализ отзывов мобильных приложений")
    
    with st.container():
        search_query = st.text_input(
            "Введите название приложения:", 
            placeholder="Например: Wildberries или TikTok",
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
    
    if st.session_state.get('selected_gp_app') or st.session_state.get('selected_ios_app'):
        display_selected_apps()

    if 'search_results' in st.session_state:
        display_search_results(st.session_state.search_results)

    if st.session_state.get('selected_gp_app') or st.session_state.get('selected_ios_app'):
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
                            if st.session_state.get('selected_gp_app'):
                                all_reviews += get_reviews(
                                    st.session_state.selected_gp_app['id'], 
                                    'google_play', 
                                    start_date, 
                                    end_date)
                            if st.session_state.get('selected_ios_app'):
                                all_reviews += get_reviews(
                                    st.session_state.selected_ios_app['id'], 
                                    'app_store', 
                                    start_date, 
                                    end_date)
                            
                            st.session_state.filtered_reviews = sorted(all_reviews, key=lambda x: x[0], reverse=True)
                            st.session_state.analysis_data = analyze_reviews(st.session_state.filtered_reviews)
                        except Exception as e:
                            st.error(f"Ошибка анализа: {str(e)}")

    if 'analysis_data' in st.session_state:
        display_analysis(
            st.session_state.analysis_data, 
            st.session_state.filtered_reviews,
            start_date,  # Передаем start_date
            end_date     # Передаем end_date
        )

if __name__ == "__main__":
    main()
