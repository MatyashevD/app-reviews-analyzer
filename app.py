import datetime
import json
import re
import requests
import spacy
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from google_play_scraper import reviews as gp_reviews, Sort
from app_store_scraper import AppStore
from collections import Counter, defaultdict
from typing import List, Dict, Any

# Инициализация NLP
nlp = spacy.load("ru_core_news_sm")

# Конфигурация DeepSeek
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
MAX_BATCH_SIZE = 5  # Уменьшено для безопасности

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
                "model": MODEL_NAME,
                "messages": [{
                    "role": "user",
                    "content": f"""
                    Проанализируй следующие отзывы. Верни JSON-массив где каждый элемент содержит:
                    - sentiment (число от 1 до 5)
                    - entities (список объектов из текста)
                    - topics (ключевые темы)
                    Отзывы: {batch}
                    """
                }],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                try:
                    parsed = json.loads(content)
                    if 'results' in parsed:
                        results.extend(parsed['results'])
                except Exception as e:
                    st.error(f"Ошибка парсинга: {str(e)}")
            
            elif response.status_code == 429:
                st.error("Превышен лимит запросов. Попробуйте позже")
                break
            
    except Exception as e:
        st.error(f"Ошибка API: {str(e)}")
    
    return results

def extract_deepseek_insights(reviews: List[tuple], deepseek_results: List[Dict]) -> Dict[str, Any]:
    insights = {
        "entities": [],
        "topics": [],
        "sentiments": [],
        "avg_sentiment": 0
    }
    
    if not deepseek_results:
        return insights
    
    # Обработка сущностей
    entity_counter = Counter()
    entity_examples = defaultdict(list)
    
    # Обработка тем
    topic_counter = Counter()
    
    # Сбор метрик
    sentiments = []
    
    for idx, result in enumerate(deepseek_results):
        # Тональность
        if 'sentiment' in result:
            sentiments.append(float(result['sentiment']))
        
        # Сущности
        for entity in result.get('entities', []):
            entity_text = entity.lower()
            entity_counter[entity_text] += 1
            if len(entity_examples[entity_text]) < 3:
                entity_examples[entity_text].append({
                    "text": reviews[idx][1],
                    "date": reviews[idx][0]
                })
        
        # Темы
        for topic in result.get('topics', []):
            topic_counter[topic.lower()] += 1
    
    # Формирование результатов
    insights['entities'] = [{
        "entity": entity,
        "count": count,
        "examples": entity_examples[entity]
    } for entity, count in entity_counter.most_common(15)]
    
    insights['topics'] = [{
        "topic": topic,
        "count": count
    } for topic, count in topic_counter.most_common(10)]
    
    if sentiments:
        insights['sentiments'] = sentiments
        insights['avg_sentiment'] = sum(sentiments) / len(sentiments)
    
    return insights

def display_deepseek_analysis(analysis: Dict[str, Any]):
    st.header("🔍 DeepSeek Analytics")
    
    if not analysis['sentiments']:
        st.warning("Данные для анализа отсутствуют")
        return
    
    # Метрики
    cols = st.columns(3)
    with cols[0]:
        st.metric("Средняя оценка", f"{analysis['avg_sentiment']:.2f}/5")
    with cols[1]:
        positive = sum(s > 3.5 for s in analysis['sentiments'])
        st.metric("Позитивные отзывы", f"{positive} ({positive/len(analysis['sentiments'])*100:.1f}%)")
    with cols[2]:
        critical = sum(s < 2.0 for s in analysis['sentiments'])
        st.metric("Критические проблемы", critical)
    
    # Сущности
    st.subheader("🏷️ Важные упоминания")
    if analysis['entities']:
        df = pd.DataFrame(analysis['entities'])
        st.dataframe(
            df[['entity', 'count']].style.bar(subset=['count'], color='#5fba7d'),
            height=400
        )
    else:
        st.info("Сущности не обнаружены")
    
    # Темы
    st.subheader("🧩 Основные темы")
    if analysis['topics']:
        topic_df = pd.DataFrame(analysis['topics'])
        st.dataframe(
            topic_df.style.background_gradient(subset=['count'], cmap='Blues'),
            hide_index=True
        )
    else:
        st.info("Темы не обнаружены")

def main():
    st.set_page_config(page_title="AI Анализатор", layout="wide")
    st.title("📱 Анализатор отзывов с DeepSeek AI")
    
    # Инициализация состояния
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    
    # Ввод данных
    col1, col2 = st.columns(2)
    with col1:
        gp_url = st.text_input("Google Play URL", help="Пример: https://play.google.com/store/apps/details?id=com.example")
    with col2:
        ios_url = st.text_input("App Store URL", help="Пример: https://apps.apple.com/ru/app/example-app/id123456789")
    
    start_date = st.date_input("Начальная дата", datetime.date(2024, 1, 1))
    end_date = st.date_input("Конечная дата", datetime.date.today())
    
    if st.button("🚀 Анализировать", type="primary"):
        with st.spinner("Собираем отзывы..."):
            gp_revs = get_google_play_reviews(gp_url)
            ios_revs = get_app_store_reviews(ios_url)
            all_reviews = gp_revs + ios_revs
            
            if not all_reviews:
                st.error("Отзывы не найдены!")
                return
                
            filtered = filter_reviews_by_date(
                all_reviews,
                datetime.datetime.combine(start_date, datetime.time.min),
                datetime.datetime.combine(end_date, datetime.time.max)
            )
            
        with st.spinner("AI-анализ..."):
            deepseek_results = analyze_with_deepseek(filtered)
            analysis = extract_deepseek_insights(filtered, deepseek_results)
            
            st.session_state.analysis_data = {
                "stats": {
                    "total": len(filtered),
                    "gp": len(gp_revs),
                    "ios": len(ios_revs)
                },
                "analysis": analysis
            }
    
    if st.session_state.analysis_data:
        data = st.session_state.analysis_data
        
        st.subheader("📊 Общая статистика")
        cols = st.columns(3)
        cols[0].metric("Всего отзывов", data['stats']['total'])
        cols[1].metric("Google Play", data['stats']['gp'])
        cols[2].metric("App Store", data['stats']['ios'])
        
        st.markdown("---")
        display_deepseek_analysis(data['analysis'])

if __name__ == "__main__":
    main()
