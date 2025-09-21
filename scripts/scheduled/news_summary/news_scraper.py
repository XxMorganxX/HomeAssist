import json,os
from datetime import datetime
from dotenv import load_dotenv
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from newspaper import Article, ArticleException

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
EPHEMERAL_DIR = os.path.join(BASE_DIR, "ephemeral_data")


class NewsHeadlineScraper:
    def __init__(self):
        self.curr_key_index = 0
        self.exhausted_keys = set()
        self.NUM_RESULTS = 3
        self.MAX_TOTAL_RESULTS = 100
        self.articles_found = 0
        
        self.switch_api()
        
    
    def switch_api(self):
        # If we're switching from a previous key, mark it as exhausted
        if self.curr_key_index > 0:
            print(f"API key {self.curr_key_index} quota exhausted")
            self.exhausted_keys.add(self.curr_key_index)
        
        # Start from the next key after the current one
        start_index = self.curr_key_index + 1
        for i in range(start_index, 6):
            if i in self.exhausted_keys:
                continue
                
            try:
                api_key = os.getenv(f"NEWS_API_KEY_{i}")
                if not api_key:
                    continue
                self.news_client = NewsApiClient(api_key=api_key)
                # Test the key with a minimal request
                test_response = self.news_client.get_top_headlines(page_size=1)
                if test_response.get('status') == 'ok':
                    self.curr_key_index = i
                    print(f"Now using API key {i}")
                    return True
            except NewsAPIException:
                print(f"API key {i} is not working, trying next...")
                self.exhausted_keys.add(i)
                continue
            except Exception as e:
                print(f"Error with API key {i}: {type(e).__name__}")
                self.exhausted_keys.add(i)
                continue
        
        # All keys exhausted
        print("All API keys have been exhausted")
        return False 

    

    def keywords_list(self):
        tech_keywords = [
            # Artificial Intelligence / Machine Learning
            "artificial intelligence", "AI", "machine learning", "deep learning",
            "generative AI", "large language models", "LLM", "neural networks",
            "computer vision", "NLP", "ChatGPT", "Claude AI", "Gemini AI",
            "openAI", "Anthropic", "stability AI",

            # Big Tech & General Tech
            "Apple", "Google", "Meta", "Amazon", "Microsoft", "Nvidia", "Tesla",
            "Alphabet", "tech layoffs", "earnings report", "big tech", "Silicon Valley",

            # Web & Software Development
            "JavaScript", "TypeScript", "React", "Node.js", "Web3", "frontend",
            "backend", "full stack", "open source", "GitHub", "API", "cloud computing",
            "DevOps", "SaaS",

            # Cybersecurity & Privacy
            "cybersecurity", "data breach", "ransomware", "phishing", "encryption",
            "VPN", "zero trust", "dark web", "malware", "cybersecurity threat",
            "information security",

            # Consumer Tech & Gadgets
            "iPhone", "Android", "smartphone", "wearable tech", "Apple Watch",
            "AR glasses", "VR headset", "Meta Quest", "foldable phone", "earbuds",
            "M3 chip", "Snapdragon",

            # Emerging Tech & Innovation
            "quantum computing", "blockchain", "augmented reality", "virtual reality",
            "mixed reality", "spatial computing", "robotics", "drones", "IoT",
            "5G", "edge computing",

            # Tech Business & Startups
            "tech IPO", "startup funding", "venture capital", "angel investing",
            "Series A", "seed round", "unicorn startup", "pitch deck", "tech accelerator",

            # Global Tech Trends
            "EU tech regulation", "China AI", "US chip ban", "semiconductors",
            "global tech summit", "CES", "WWDC", "MWC", "tech trade war"
        ]
        return tech_keywords


    def update_headlines(self, new_headlines: list, searched_keywords=None):
        headlines_dir = EPHEMERAL_DIR
        headlines_path = os.path.join(headlines_dir, "headlines.json")
        os.makedirs(headlines_dir, exist_ok=True)
        # Load existing headlines or start with empty list
        if os.path.exists(headlines_path):
            try:
                with open(headlines_path, "r") as f:
                    data = json.load(f)
                    if not isinstance(data, list) or not data:
                        metadata = {}
                        headlines_in_file = []
                    else:
                        metadata = data[0] if isinstance(data[0], dict) else {}
                        headlines_in_file = data[1:] if len(data) > 1 else []
            except Exception:
                metadata = {}
                headlines_in_file = []
        else:
            metadata = {}
            headlines_in_file = []
        # Update metadata
        now = datetime.now().isoformat()
        # Track all searched keywords and found keywords
        if searched_keywords is None:
            searched_keywords = metadata.get("searched_keywords", [])
        found_keywords = set(metadata.get("found_keywords", []))
        # Add any new found keywords from new_headlines
        for h in new_headlines:
            if isinstance(h, dict) and "keyword" in h:
                found_keywords.add(h["keyword"])
        total_articles_found = len(headlines_in_file) + len(new_headlines)
        metadata = {
            "last_added": now,
            "searched_keywords": list(set(searched_keywords)),
            "found_keywords": list(found_keywords),
            "total_articles_found": total_articles_found
        }
        # Write metadata as first element, then all headlines
        all_data = [metadata] + headlines_in_file + new_headlines
        with open(headlines_path, "w") as f:
            json.dump(all_data, f, indent=2)

    def fetch_headlines(self, country: str = None, category: str = None, sources: str = None, page_size: int = 20):
        """
        Fetch top headlines matching the given keywords and filters.
        """
        headlines_dir = EPHEMERAL_DIR
        headlines_path = os.path.join(headlines_dir, "headlines.json")
        # Load searched_keywords from headlines.json if it exists
        already_searched = set()
        if os.path.exists(headlines_path):
            try:
                with open(headlines_path, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        already_searched = set(data[0].get("searched_keywords", []))
                        existing_articles = data[0].get("total_articles_found", 0)
                        if already_searched:
                            print(f"\n{len(already_searched)} keywords already searched, {existing_articles} articles already collected")
                            print("Continuing with remaining keywords...\n")
            except Exception:
                pass
        searched_keywords = []
        keywords = self.keywords_list()
        keyword_index = 0
        
        while keyword_index < len(keywords):
            keyword = keywords[keyword_index]
            
            if self.articles_found >= self.MAX_TOTAL_RESULTS:
                break
            if keyword in already_searched:
                keyword_index += 1
                continue
                
            print(f"Fetching headlines for '{keyword}'")
            try:
                top_headlines = self.news_client.get_top_headlines(
                    q=keyword,
                    country="us",
                    category="technology",
                    page_size=self.NUM_RESULTS,
                    page=1
                )
                
                articles = top_headlines.get("articles", [])
                total_results = top_headlines.get("totalResults", 0)
                
                if articles:
                    print(f"  ✓ Found {len(articles)} articles (total available: {total_results})")
                    # Add the keyword and fetched_at timestamp to each article
                    fetched_time = datetime.now().isoformat()
                    for i, article in enumerate(articles, 1):
                        article["keyword"] = keyword
                        article["fetched_at"] = fetched_time
                        if "urlToImage" in article:
                            del article["urlToImage"]
                        # Show article titles for transparency
                        title = article.get('title', 'No title')[:60] + "..." if len(article.get('title', '')) > 60 else article.get('title', 'No title')
                        print(f"    {i}. {title}")
                    
                    self.articles_found += len(articles)
                    self.update_headlines(articles, searched_keywords=searched_keywords + list(already_searched))
                else:
                    print(f"  ✗ No articles found for '{keyword}'")
                
                # Successfully processed this keyword, add to searched list
                searched_keywords.append(keyword)
                keyword_index += 1
                
            except NewsAPIException as e:
                print(f"  API quota exceeded: {str(e)}")
                # Quota exceeded - try next API key
                if not self.switch_api():
                    # All API keys exhausted - save what we have and exit gracefully
                    print(f"\nAll API keys exhausted. Total articles fetched: {self.articles_found}")
                    return
                # Retry the same keyword with new API key (don't increment keyword_index)
                print(f"  Retrying '{keyword}' with new API key...")
    
    
    def get_num_results_found(self):
        path = os.path.join(EPHEMERAL_DIR, "headlines.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "r") as f:
            data = json.load(f)
            metadata = data[0]
            print(f"\n=== Scraping Summary ===")
            print(f"Total articles found: {metadata['total_articles_found']}")
            print(f"Keywords searched: {len(metadata.get('searched_keywords', []))}")
            print(f"Keywords with results: {len(metadata.get('found_keywords', []))}")
            
            if metadata.get('found_keywords'):
                print(f"\nSuccessful keywords: {', '.join(metadata['found_keywords'])}")


class NewsScraper:
    def __init__(self, urls):
        self.urls = urls
        self.scrape_news()
        self.display_news = False

    def scrape_news(self):
        self.articles = []
        for i, url in enumerate(self.urls, 1):
            print(f"\n{'='*50}")
            print(f"Article {i}: Fetching content from {url[:40]}")
            
            
            try:
                article = Article(url)
                article.download()
                article.parse()
                self.articles.append(article)
                
            except ArticleException as e:
                print(f"\n❌ {type(e).__name__} error fetching article")
            
            print('='*50)

    def store_news(self):
        for article in self.articles:
            if self.display_news:
                print("="*50)
                print("Title: ", article.title)
                print("Text: ", article.text)
                print("="*50)
        
        dump_articles = {}
        for i in range(len(self.articles)):
            article = self.articles[i]
            dump_articles[str(i+1)] = {
                "title": article.title,
                "text": article.text,
                "authors": article.authors,
                "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                "url": article.url
            }
        out_path = os.path.join(EPHEMERAL_DIR, "news.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dump_articles, f, indent=2)

    def get_news(self):
        path = os.path.join(EPHEMERAL_DIR, "news.json")
        with open(path, "r") as f:
            self.articles = json.load(f)
        return self.articles
    


if __name__ == "__main__":  
    news_fetcher = NewsHeadlineScraper()
    news_fetcher.fetch_headlines()
    news_fetcher.get_num_results_found()