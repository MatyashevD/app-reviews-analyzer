import datetime
import spacy
import pandas as pd
import re
import streamlit as st
import matplotlib.pyplot as plt
import requests
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter, defaultdict
from typing import List, Dict, Any

nlp = spacy.load("ru_core_news_sm")

# DeepSeek API конфигурация
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/analyze"
MAX_BATCH_SIZE = 10  # Для соблюдения лимитов API

def extract_google_play_id(url: str) -> str:
    match = re.search(r'id=([a-zA-Z0-9._-]+)', url)
    return match.group(1) if match else None

def extract_app_store_id(url: str) -> str:
    match = re.search(r'/id(\d+)', url)
    return match.group(1) if match else None

def get_google_play_reviews(app_url: str, lang: str = 'ru', country: str = 'ru', count: int = 100) -> List[tuple]:
    app_id = extract_google_play_id(app_url)
    if not app_id:
        st.warning("Неверный URL Google Play")
        return []
    
    try:
        result, _ = gp_reviews(
            app_id,
            lang=lang,
            country=country,
            count=count,
            sort=Sort.NEWEST
        )
        return [(r['at'], r['content'], 'Google Play') for r in result]
    except Exception as e:
        st.error(f"Ошибка Google Play: {str(e)}")
        return []

def get_app_store_reviews(app_url: str, country: str = 'ru', count: int = 100) -> List[tuple]:
    app_id = extract_app_store_id(app_url)
    if not app_id:
        st.warning("Неверный URL App Store")
        return []
    
    try:
        app_name_match = re.search(r'/app/([^/]+)/', app_url)
        app_name = app_name_match.group(1) if app_name_match else "unknown_app"
        
        app = AppStore(country=country, app_id=app_id, app_name=app_name)
        app.review(how_many=count)
        
        return [(r['date'], r['review'], 'App Store') for r in app.reviews]
    except Exception as e:
        st.error(f"Ошибка App Store: {str(e)}")
        return []

def filter_reviews_by_date(reviews: List[tuple], start_date: datetime.datetime, end_date: datetime.datetime) -> List[tuple]:
    return [r for r in reviews if start_date <= r[0] <= end_date]

def analyze_with_deepseek(reviews: List[tuple]) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {st.secrets['DEEPSEEK_API_KEY']}",
        "Content-Type": "application/json"
    }
    
    results = []
    texts = [review[1] for review in reviews]
    
    try:
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i:i+MAX_BATCH_SIZE]
            payload = {
                "texts": batch,
                "features": ["sentiment", "entities", "topics", "keywords"],
                "language": "ru"
            }
            
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            results.extend(response.json()['results'])
            
        return results
    except Exception as e:
        st.error(f"Ошибка DeepSeek API: {str(e)}")
        return []

def extract_deepseek_insights(reviews: List[tuple], deepseek_results: List[Dict]) -> Dict[str, Any]:
    if not deepseek_results:
        return {}
    
    # Анализ сущностей
    entity_counter = Counter()
    entity_examples = defaultdict(list)
    
    # Анализ тем
    topic_counter = Counter()
    
    # Сбор метрик
    sentiments = []
    
    for idx, result in enumerate(deepseek_results):
        # Обработка сущностей
        for entity in result.get('entities', []):
            entity_text = entity['text'].lower()
            entity_counter[entity_text] += 1
            if len(entity_examples[entity_text]) < 3:
                entity_examples[entity_text].append({
                    "text": reviews[idx][1],
                    "sentiment": result['sentiment'],
                    "date": reviews[idx][0]
                })
        
        # Обработка тем
        for topic in result.get('topics', []):
            topic_counter[topic] += 1
        
        # Сбор тональности
        sentiments.append(result.get('sentiment', 3.0))
    
    # Подготовка данных
    top_entities = [{
        "entity": entity,
        "count": count,
        "sentiment": sum(e['sentiment'] for e in entity_examples[entity])/len(entity_examples[entity]),
        "examples": entity_examples[entity]
    } for entity, count in entity_counter.most_common(15)]
    
    top_topics = [{
        "topic": topic,
        "count": count,
        "keywords": ", ".join(result.get('keywords', [])[:5])
    } for topic, count in topic_counter.most_common(10)]
    
    return {
        "entities": top_entities,
        "topics": top_topics,
        "sentiments": sentiments,
        "avg_sentiment": sum(sentiments)/len(sentiments) if sentiments else 0
    }

def display_deepseek_analysis(analysis: Dict[str, Any]):
    st.header("🔍 DeepSeek Advanced Analytics")
    
    # Метрики
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Средняя тональность", f"{analysis['avg_sentiment']:.2f}/5")
    with col2:
        positive = sum(1 for s in analysis['sentiments'] if s > 3.5)
        st.metric("Позитивные отзывы", f"{positive} ({positive/len(analysis['sentiments'])*100:.1f}%)")
    with col3:
        critical = sum(1 for s in analysis['sentiments'] if s < 2.0)
        st.metric("Критические проблемы", critical)
    
    # Сущности
    st.subheader("🏷️ Ключевые сущности")
    entity_df = pd.DataFrame(analysis['entities'])
    if not entity_df.empty:
        st.dataframe(
            entity_df[['entity', 'count', 'sentiment']]
            .style.background_gradient(subset=['sentiment'], cmap='RdYlGn', vmin=1, vmax=5)
            .format({'sentiment': "{:.2f}"}),
            height=400
        )
    else:
        st.info("Сущности не обнаружены")
    
    # Темы
    st.subheader("🧩 Автоматические темы")
    topic_df = pd.DataFrame(analysis['topics'])
    if not topic_df.empty:
        st.dataframe(
            topic_df.style.bar(subset=['count'], color='#5fba7d'),
            column_config={
                "keywords": st.column_config.ListColumn(
                    width="large",
                    help="Связанные ключевые слова"
                )
            }
        )
    else:
        st.info("Темы не обнаружены")

def main():
    st.set_page_config(page_title="AI Анализатор отзывов", layout="wide")
    st.title("📱 AI Анализ отзывов")
    
    # Инициализация сессии
    session_defaults = {
        'analysis_data': None,
        'selected_word': None,
        'active_tab': "Топ упоминаний"
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Ввод данных
    col1, col2 = st.columns(2)
    with col1:
        gp_url = st.text_input("Ссылка Google Play", placeholder="https://play.google.com/store/apps/details?id=...")
    with col2:
        ios_url = st.text_input("Ссылка App Store", placeholder="https://apps.apple.com/ru/app/...")
    
    start_date = st.date_input("Начальная дата", datetime.date(2024, 1, 1))
    end_date = st.date_input("Конечная дата", datetime.date.today())
    
    if st.button("🚀 Запустить анализ", type="primary"):
        st.session_state.analysis_data = None
        start_dt = datetime.datetime.combine(start_date, datetime.time.min)
        end_dt = datetime.datetime.combine(end_date, datetime.time.max)
        
        with st.spinner("🕸️ Сбор отзывов..."):
            gp_revs = get_google_play_reviews(gp_url)
            ios_revs = get_app_store_reviews(ios_url)
        
        all_reviews = gp_revs + ios_revs
        if not all_reviews:
            st.error("Отзывы не найдены!")
            return
            
        filtered_reviews = filter_reviews_by_date(all_reviews, start_dt, end_dt)
        
        with st.spinner("🤖 Анализ с DeepSeek..."):
            deepseek_results = analyze_with_deepseek(filtered_reviews)
            deepseek_analysis = extract_deepseek_insights(filtered_reviews, deepseek_results)
            
            st.session_state.analysis_data = {
                'basic': {
                    'reviews': filtered_reviews,
                    'gp_count': len([r for r in filtered_reviews if r[2] == 'Google Play']),
                    'ios_count': len([r for r in filtered_reviews if r[2] == 'App Store'])
                },
                'deepseek': deepseek_analysis
            }

    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        
        # Общая информация
        st.subheader("📊 Основные метрики")
        cols = st.columns(4)
        cols[0].metric("Всего отзывов", len(data['basic']['reviews']))
        cols[1].metric("Google Play", data['basic']['gp_count'])
        cols[2].metric("App Store", data['basic']['ios_count'])
        cols[3].metric("Качество анализа", 
                      f"{len(data['deepseek']['sentiments'])/len(data['basic']['reviews'])*100:.1f}%")
        
        # Вкладки
        tabs = ["📌 Топ упоминаний", "🧭 Ручные темы", "🔍 Детализация", "🤖 DeepSeek AI"]
        st.session_state.active_tab = st.radio(
            "Режимы анализа:",
            tabs,
            index=tabs.index(st.session_state.active_tab),
            horizontal=True,
            label_visibility="collapsed"
        )

        if st.session_state.active_tab == "🤖 DeepSeek AI":
            display_deepseek_analysis(data['deepseek'])
            
            # Дополнительные графики
            st.subheader("📈 Динамика тональности")
            sentiment_df = pd.DataFrame({
                "Дата": [r[0] for r in data['basic']['reviews']],
                "Тональность": data['deepseek']['sentiments']
            })
            st.line_chart(sentiment_df.set_index('Дата')['Тональность'])
            
        # Остальные вкладки можно дополнить аналогично...

if __name__ == "__main__":
    main()
