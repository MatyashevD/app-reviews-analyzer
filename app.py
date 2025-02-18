import streamlit as st
import pandas as pd
from textblob import TextBlob
from google_play_scraper import app, search, Sort, reviews_all
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

def parse_google_play_url(url):
    try:
        parsed = urlparse(url)
        if not parsed.netloc == 'play.google.com':
            return None
        path_parts = parsed.path.split('/')
        if len(path_parts) >= 4 and path_parts[1] == 'store':
            return path_parts[3]
        return None
    except:
        return None

def get_app_reviews(app_ids, review_count=100, lang='en', country='us'):
    all_reviews = []
    for app_id in app_ids:
        result, continuation_token = reviews_all(
            app_id,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=review_count,
            filter_score_with=None
        )
        for review in result:
            all_reviews.append({
                'App ID': app_id,
                'User': review['userName'],
                'Rating': review['score'],
                'Review': review['content'],
                'Date': review['at'].strftime('%Y-%m-%d'),
                'Likes': review['thumbsUpCount'],
                'Sentiment': analyze_sentiment(review['content'])
            })
    return pd.DataFrame(all_reviews)

# Интерфейс Streamlit
st.title('📱 Advanced App Review Analyzer')
st.markdown("""
Анализируйте отзывы приложений из Google Play Store:
1. Поиск по названию приложения или ввод прямых ссылок
2. Анализ тональности отзывов
3. Сравнение нескольких приложений
4. Экспорт результатов в различных форматах
""")

# Выбор режима ввода
input_method = st.radio("Выберите метод ввода:", ['Поиск по названию', 'Ввод ссылок вручную'])

app_ids = []
app_names = []

if input_method == 'Поиск по названию':
    search_query = st.text_input('Введите название приложения для поиска:')
    country = st.selectbox('Выберите страну:', ['us', 'ru', 'de', 'fr', 'jp', 'kr'])
    lang = st.selectbox('Выберите язык:', ['en', 'ru', 'de', 'fr', 'ja', 'ko'])
    
    if search_query:
        search_results = search(search_query, lang=lang, country=country)
        if search_results:
            selected_apps = st.multiselect(
                'Выберите приложения:',
                options=[f"{app['title']} ({app['appId']})" for app in search_results],
                format_func=lambda x: x.split(' (')[0]
            )
            app_ids = [app.split(' (')[1][:-1] for app in selected_apps]
            app_names = [app.split(' (')[0] for app in selected_apps]
        else:
            st.warning('Приложения по вашему запросу не найдены.')

else:
    urls = st.text_area('Введите ссылки на приложения (по одной в строке):')
    if urls:
        urls_list = urls.split('\n')
        app_ids = []
        for url in urls_list:
            url = url.strip()
            if url:
                app_id = parse_google_play_url(url)
                if app_id:
                    app_ids.append(app_id)
                    try:
                        app_info = app(app_id)
                        app_names.append(app_info['title'])
                    except:
                        app_names.append(app_id)
                else:
                    st.error(f'Некорректная ссылка: {url}')

if app_ids:
    st.success(f'Выбрано приложений: {len(app_ids)}')
    review_count = st.slider('Количество отзывов для анализа:', 50, 1000, 200, 50)
    
    if st.button('Анализировать отзывы'):
        with st.spinner('Собираем отзывы... Это может занять несколько минут'):
            reviews_df = get_app_reviews(app_ids, review_count, lang, country)
        
        if not reviews_df.empty:
            # Статистика
            st.subheader('📊 Общая статистика')
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Средний рейтинг", f"{reviews_df['Rating'].mean():.1f} ⭐")
            
            with col2:
                sentiment_dist = reviews_df['Sentiment'].value_counts(normalize=True).mul(100)
                st.write("**Распределение тональности:**")
                for sentiment, percent in sentiment_dist.items():
                    st.write(f"{sentiment}: {percent:.1f}%")
            
            with col3:
                rating_dist = reviews_df['Rating'].value_counts().sort_index(ascending=False)
                st.write("**Распределение оценок:**")
                for rating, count in rating_dist.items():
                    st.write(f"{'⭐' * int(rating)}: {count} отзывов")

            # Детализированные данные
            st.subheader('📝 Детализированные отзывы')
            st.dataframe(reviews_df.style.background_gradient(cmap='Blues', subset=['Rating', 'Likes']), 
                        height=500,
                        use_container_width=True)

            # Экспорт данных
            st.subheader('📤 Экспорт данных')
            
            csv = reviews_df.to_csv(index=False).encode('utf-8')
            excel = reviews_df.to_excel(index=False, engine='openpyxl')
            
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
