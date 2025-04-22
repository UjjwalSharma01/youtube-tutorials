import feedparser
from flask import Flask, render_template, request
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import requests
from bs4 import BeautifulSoup
import time
import re

app = Flask(__name__)

RSS_FEEDS = {
    'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
    'Hacker News': 'https://news.ycombinator.com/rss',
    'Wall Street Journal': 'https://feeds.a.dj.com/rss/RSSMarketsMain.xml',
    'CNBC': 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069'
}

# Function to extract article content
def extract_article_content(url, timeout=5):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
                script_or_style.decompose()
            
            # Extract text from paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 50])
            
            # If no substantial paragraphs found, try getting all text
            if not text:
                text = soup.get_text()
                # Clean up the text (remove extra whitespace, etc.)
                text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        return None
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return None

# Function to generate a summary from text
def generate_summary_from_text(text, language="english", sentences_count=3):
    if not text or len(text.split()) < 20:  # Need at least 20 words for meaningful summary
        return None
        
    try:
        # Create parser and tokenize
        parser = PlaintextParser.from_string(text, Tokenizer(language))
        
        # Use LexRank summarizer (better than LSA for news articles)
        stemmer = Stemmer(language)
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)
        
        # Get sentences for summary
        sentences = summarizer(parser.document, sentences_count)
        
        if sentences:
            return " ".join(str(sentence) for sentence in sentences)
        return None
    except Exception as e:
        print(f"Error in summarization: {e}")
        return None

# Main function to get article summary
def get_article_summary(entry):
    # Start with entry content if available (may contain full text in some feeds)
    if hasattr(entry, 'content') and entry.content:
        for content in entry.content:
            if hasattr(content, 'value') and content.value:
                text = content.value
                summary = generate_summary_from_text(text)
                if summary:
                    return summary
    
    # Try using description if available
    if hasattr(entry, 'description') and entry.description:
        # Strip HTML tags from description
        text = BeautifulSoup(entry.description, 'html.parser').get_text()
        if len(text.split()) > 20:  # Only if description has enough content
            summary = generate_summary_from_text(text)
            if summary and summary.lower() != entry.title.lower():
                return summary
    
    # If no summary yet, fetch the actual article content
    try:
        article_text = extract_article_content(entry.link)
        if article_text:
            summary = generate_summary_from_text(article_text)
            if summary and summary.lower() != entry.title.lower():
                return summary
    except Exception as e:
        print(f"Error fetching article content: {e}")
    
    # Return a message if we couldn't generate a proper summary
    return "No detailed summary available for this article."


@app.route('/')
def index():
    articles = []
    for source, feed in RSS_FEEDS.items():
        parsed_feed = feedparser.parse(feed)
        entries = []
        for entry in parsed_feed.entries:
            # Get summary for the article (with a small delay to avoid overwhelming servers)
            summary = get_article_summary(entry)
            entries.append((source, entry, summary))
            time.sleep(0.1)  # Small delay between requests
        articles.extend(entries)

    articles = sorted(articles, key=lambda x: x[1].published_parsed, reverse=True)

    page = request.args.get('page', 1, type=int)
    per_page = 10
    total_articles = len(articles)
    start = (page-1) * per_page
    end = start + per_page
    paginated_articles = articles[start:end]

    return render_template('index.html', articles=paginated_articles, page=page,
                           total_pages = total_articles // per_page + 1)


@app.route('/search')
def search():
    query = request.args.get('q')

    articles = []
    for source, feed in RSS_FEEDS.items():
        parsed_feed = feedparser.parse(feed)
        entries = []
        for entry in parsed_feed.entries:
            # Get summary for each article
            summary = get_article_summary(entry)
            entries.append((source, entry, summary))
            time.sleep(0.1)  # Small delay between requests
        articles.extend(entries)

    results = [article for article in articles if query.lower() in article[1].title.lower()]

    return render_template('search_results.html', articles=results, query=query)


if __name__ == '__main__':
    app.run(debug=True)
