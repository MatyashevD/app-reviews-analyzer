import streamlit as st
import pandas as pd
from textblob import TextBlob
from google_play_scraper import app as gp_app, search as gp_search, Sort, reviews_all as gp_reviews_all
from app_store_scraper import AppStore
from urllib.parse import urlparse
import warnings
import re

# Настройка подавления предупреждений
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    if analysis.sentiment.polarity > 0.2:
        return 'Positive'
    elif analysis.sentiment.polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'

def parse_store_url(url, store):
    try:
        parsed = urlparse(url)
        if store == 'google_play':
            if not parsed.netloc == 'play.google.com':
                return None
            path_parts = parsed.path.split('/')
            if len(path_parts) >= 4 and path_parts[1] == 'store':
                return path_parts[3]
        elif store == 'app_store':
            if not parsed.netloc == 'apps.apple.com':
                return None
            path_parts = parsed.path.split('/')
            if len(path_parts) >= 4 and path_parts[1] == 'app':
                return path_parts[3]
        return None
    except:
        return None

def get_reviews(app_info, store, review_count):
    all_reviews = []
    
    if store == 'google_play':
        result = gp_reviews_all(
            app_info['id'],
            lang='ru',
            country='ru',
            sort=Sort.NEWEST,
            count=review_count,
            filter_score_with=None
        )
        for review in result:
            all_reviews.append({
                'Store': 'Google Play',
                'App Name': app_info['name'],
                'User': review['userName'],
                'Rating': review['score'],
                'Review': review['content'],
                'Date': review['at'].strftime('%Y-%m-%d'),
                'Likes': review['thumbsUpCount'],
                'Sentiment': analyze_sentiment(review['content'])
            })
            
    elif store == 'app_store':
        app = AppStore(country='ru', app_name=app_info['name'], app_id=app_info['id'])
        app.review(how_many=review_count)
        
        for review in app.reviews:
            all_reviews.append({
                'Store': 'App Store',
                'App Name': app_info['name'],
                'User': review['userName'],
                'Rating': review['rating'],
                'Review': review['review'],
                'Date': review['date'].strftime('%Y-%m-%d'),
                'Likes': review.get('thumbsUpCount', 0),
                'Sentiment': analyze_sentiment(review['review'])
            })
    
    return pd.DataFrame(all_reviews)

# Интерфейс Streamlit
st.title('📱 Universal App Review Analyzer')
st.markdown("""
Анализируйте отзывы приложений из:
1. Google Play Store
2. Apple App Store

Функционал:
- Поиск по названию или ссылкам
- Анализ тональности отзывов
- Сравнение приложений
- Экспорт результатов
""")

# Выбор магазина
store = st.radio("Выберите магазин:", ['Google Play', 'App Store']).lower().replace(' ', '_')

# Выбор режима ввода
input_method = st.radio("Выберите метод ввода:", ['Поиск по названию', 'Ввод ссылок вручную'])

apps_info = []

if input_method == 'Поиск по названию':
    search_query = st.text_input('Введите название приложения для поиска:')
    
    if search_query:
        if store == 'google_play':
            search_results = gp_search(search_query, lang='ru', country='ru')
            if search_results:
                selected_apps = st.multiselect(
                    'Выберите приложения:',
                    options=[f"{app['title']} ({app['appId']})" for app in search_results],
                    format_func=lambda x: x.split(' (')[0]
                )
                for app in selected_apps:
                    app_id = app.split(' (')[1][:-1]
                    try:
                        app_details = gp_app(app_id)
                        apps_info.append({'id': app_id, 'name': app_details['title']})
                    except:
                        apps_info.append({'id': app_id, 'name': app_id})
        elif store == 'app_store':
            st.info('Для App Store используйте ручной ввод ссылок. Поиск по названию недоступен.')

else:
    urls = st.text_area('Введите ссылки на приложения (по одной в строке):')
    if urls:
        urls_list = urls.split('\n')
        for url in urls_list:
            url = url.strip()
            if url:
                app_id = parse_store_url(url, store)
                if app_id:
                    try:
                        if store == 'google_play':
                            app_details = gp_app(app_id)
                            name = app_details['title']
                        elif store == 'app_store':
                            name = "Название приложения"  # Для App Store имя получаем через API
                        apps_info.append({'id': app_id, 'name': name})
                    except:
                        apps_info.append({'id': app_id, 'name': app_id})
                else:
                    st.error(f'Некорректная ссылка: {url}')

if apps_info:
    st.success(f'Выбрано приложений: {len(apps_info)}')
    review_count = st.slider('Количество отзывов для анализа:', 50, 1000, 200, 50)
    
    if st.button('Анализировать отзывы'):
        all_reviews = pd.DataFrame()
        with st.spinner('Собираем отзывы... Это может занять несколько минут'):
            for app in apps_info:
                reviews_df = get_reviews(app, store, review_count)
                if not reviews_df.empty:
                    all_reviews = pd.concat([all_reviews, reviews_df], ignore_index=True)
        
        if not all_reviews.empty:
            # Статистика
            st.subheader('📊 Общая статистика')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Средний рейтинг", f"{all_reviews['Rating'].mean():.1f} ⭐")
            
            with col2:
                sentiment_dist = all_reviews['Sentiment'].value_counts(normalize=True).mul(100)
                st.write("**Распределение тональности:**")
                for sentiment, percent in sentiment_dist.items():
                    st.write(f"{sentiment}: {percent:.1f}%")
            
            with col3:
                rating_dist = all_reviews['Rating'].value_counts().sort_index(ascending=False)
                st.write("**Распределение оценок:**")
                for rating, count in rating_dist.items():
                    st.write(f"{'⭐' * int(rating)}: {count} отзывов")

            # Детализированные данные
            st.subheader('📝 Детализированные отзывы')
            st.dataframe(all_reviews.style.background_gradient(cmap='Blues', subset=['Rating', 'Likes']), 
                        height=500,
                        use_container_width=True)

            # Экспорт данных
            st.subheader('📤 Экспорт данных')
            
            csv = all_reviews.to_csv(index=False).encode('utf-8')
            excel = all_reviews.to_excel(index=False, engine='openpyxl')
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Скачать CSV",
                    data=csv,
                    file_name='app_reviews.csv',
                    mime='text/csv'
                )
            with col2:
                st.download_button(
                    label="Скачать Excel",
                    data=excel,
                    file_name='app_reviews.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        else:
            st.warning('Не удалось получить отзывы для выбранных приложений.')
else:
    st.info('👆 Выберите приложения для анализа')

st.markdown("---")
st.caption("Инструмент разработан для комплексного анализа отзывов мобильных приложений")
