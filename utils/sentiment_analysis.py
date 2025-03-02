def process_and_rank_news(news_articles):
    ranked_news = []

    for article in news_articles:
        title = article['title']
        publishedAt = article['publishedAt']
        
        # Sentiment analysis using TextBlob
        sentiment_analysis = TextBlob(title)
        sentiment = 'Positive' if sentiment_analysis.sentiment.polarity > 0 else 'Negative'

        # Randomly assign an impact score for demo (you can replace this with real calculations)
        impact_score = round(abs(sentiment_analysis.sentiment.polarity) * 10, 2)

        # Investment action based on sentiment (simple rule-based)
        if sentiment == 'Positive':
            investment_action = 'Buy'
        else:
            investment_action = 'Sell'

        # Store the result for ranking
        ranked_news.append({
            'title': title,
            'publishedAt': publishedAt,
            'sentiment': sentiment,
            'impact_score': impact_score,
            'investment_action': investment_action
        })

    # Sort the ranked news based on impact score (descending)
    ranked_news = sorted(ranked_news, key=lambda x: x['impact_score'], reverse=True)

    print(f"Ranked news: {ranked_news}")  # Debugging line to check the ranked news

    return ranked_news
