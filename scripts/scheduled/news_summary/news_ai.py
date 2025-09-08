import json,os
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv
from newspaper import Article, ArticleException
from news_scraper import NewsScraper

load_dotenv()


class BaseAnalysis(ABC):
    """Abstract base class for all analysis types"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
        self.model = "gpt-4o-mini"
        self.max_tokens = 1000
        
    @abstractmethod
    def init_openai_params(self):
        """Initialize OpenAI parameters including prompt and messages"""
        pass
        
    @abstractmethod
    def analyze(self):
        """Perform the analysis"""
        pass
        
    def call_openai(self, messages, response_format=None):
        """Common OpenAI API call logic"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens
            }
            
            if response_format:
                kwargs["response_format"] = response_format
                
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {type(e).__name__}: {str(e)}")
            return None
            
    def save_results(self, data, filepath):
        """Save analysis results to JSON file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Analysis saved to {filepath}")
            return True
        except Exception as e:
            print(f"Error saving results: {type(e).__name__}: {str(e)}")
            return False


class HeadlineAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()
        self.output_filepath = "ephemeral_data/top_headlines.json"
        
        if os.path.exists(self.output_filepath):
            print("Headlines already analyzed. Skipping...")
            return
        
        self.headlines = self.get_headlines()
        if not self.headlines:
            print("No headlines found to analyze")
            return
            
        self.init_openai_params()
        print(f"Analyzing {len(self.headlines)} headlines...")
        for headline in self.headlines:
            print(f"- {headline['title']}")
        
        self.analyze()

    def init_openai_params(self):
        self.prompt = """
            You are a news analyst. You are given a list of headlines.
            Your job is to determine the 5 best headlines from the list.
            The best headline are the ones that are most relevant to technical innovation in AI and the tech industry.

            You must output your response in valid JSON format with the following structure:
            {
                "1": {
                    "headline": "[exact headline text]",
                    "url": "[article URL]",
                    "reasoning": "[your reasoning here]"
                },
                "2": {
                    "headline": "[exact headline text]",
                    "url": "[article URL]",
                    "reasoning": "[your reasoning here]"
                },
                "3": {
                    "headline": "[exact headline text]",
                    "url": "[article URL]",
                    "reasoning": "[your reasoning here]"
                },
                "4": {
                    "headline": "[exact headline text]",
                    "url": "[article URL]",
                    "reasoning": "[your reasoning here]"
                },
                "5": {
                    "headline": "[exact headline text]",
                    "url": "[article URL]",
                    "reasoning": "[your reasoning here]"
                }
            }
            
            The reasoning is a short explanation of why the headline is the best.
            The reasoning should be at least 100 words.
            Remember to output valid JSON only.
        """

        # Convert headlines to a formatted string with URLs
        headlines_text = "\n".join([f"{i+1}. {h['title']} - URL: {h['url']}" for i, h in enumerate(self.headlines)])
        
        self.messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": f"Here are the headlines:\n{headlines_text}"}
        ]
        

    def get_headlines(self):
        if not os.path.exists("ephemeral_data/headlines.json"):
            return []
        with open("ephemeral_data/headlines.json", "r") as f:
            self.headlines = json.load(f)
        
        return self.headlines[1:]

    def analyze(self):
        """Analyze headlines and determine the best ones"""
        result = self.call_openai(self.messages, response_format={"type": "json_object"})
        
        if not result:
            print("Failed to get analysis from OpenAI")
            return None
        
        # Parse and display the JSON response
        try:
            parsed_result = json.loads(result)
            print("\n=== Top 5 Headlines Analysis ===\n")
            
            for rank, data in parsed_result.items():
                print(f"#{rank}: {data['headline']}")
                if 'url' in data:
                    print(f"URL: {data['url']}")
                print(f"Reasoning: {data['reasoning']}\n")
                
            # Save the analysis
            self.save_results(parsed_result, self.output_filepath)
            
        except json.JSONDecodeError:
            print("Error parsing JSON response")
            print(result)
            return None
            
        return parsed_result
    
    def get_urls(self):
        data = None
        urls = []
        with open("ephemeral_data/top_headlines.json", "r") as f:
            data = json.load(f)

        for key, value in data.items():
            if key in ['1', '2', '3', '4', '5']:
                urls.append(value['url'])
        return urls


class FullArticleAnalysis(BaseAnalysis):
    """Analyze full article content for technical innovation and insights"""
    
    def __init__(self):
        super().__init__()
        self.output_filepath = "ephemeral_data/full_article_analysis.json"
        self.news_filepath = "ephemeral_data/news.json"
        self.max_tokens = 2000  # Increase for more detailed analysis
        
        if os.path.exists(self.output_filepath):
            print("Articles already analyzed. Skipping...")
            return
            
        self.articles = self.load_articles()
        if not self.articles:
            print("No articles found to analyze")
            return
            
        print(f"\nAnalyzing {len(self.articles)} full articles...")
        self.analyze_all_articles()
        
    def load_articles(self):
        """Load scraped articles from news.json"""
        if not os.path.exists(self.news_filepath):
            print(f"No articles found at {self.news_filepath}")
            return None
            
        try:
            with open(self.news_filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading articles: {type(e).__name__}: {str(e)}")
            return None
            
    def init_openai_params(self, article_data):
        """Initialize OpenAI parameters for a single article"""
        self.prompt = """
            You are a technology analyst specializing in AI and innovation. 
            Analyze this article and provide a comprehensive assessment.
            
            You must output your response in valid JSON format with the following structure:
            {
                "title": "[article title]",
                "url": "[article URL]",
                "innovation_score": [float between 1-10],
                "key_insights": ["insight1", "insight2", "insight3"],
                "tech_impact": "[description of technological impact]",
                "ai_relevance": "[specific relevance to AI/ML if applicable]",
                "summary": "[comprehensive 150-200 word summary]",
                "categories": ["category1", "category2"],
                "future_implications": "[potential future developments]"
            }
            
            Innovation Score Guidelines:
            - 1-3: Minor incremental improvements
            - 4-6: Significant feature additions or improvements
            - 7-8: Major breakthrough or paradigm shift
            - 9-10: Revolutionary innovation with industry-wide impact
            
            Categories can include: AI, Machine Learning, Hardware, Software, 
            Consumer Tech, Enterprise, Security, Research, etc.
            
            Focus on technical innovation, practical applications, and industry impact.
        """
        
        article_content = f"""
        Title: {article_data.get('title', 'No title')}
        URL: {article_data.get('url', 'No URL')}
        
        Article Text:
        {article_data.get('text', 'No content available')[:3000]}
        """
        
        self.messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": article_content}
        ]
        
    def analyze_single_article(self, article_key, article_data):
        """Analyze a single article"""
        print(f"\nAnalyzing article {article_key}: {article_data.get('title', 'Unknown')[:60]}...")
        
        self.init_openai_params(article_data)
        result = self.call_openai(self.messages, response_format={"type": "json_object"})
        
        if not result:
            print(f"Failed to analyze article {article_key}")
            return None
            
        try:
            parsed_result = json.loads(result)
            
            # Display key findings
            print(f"  Innovation Score: {parsed_result.get('innovation_score', 'N/A')}/10")
            print(f"  Categories: {', '.join(parsed_result.get('categories', []))}")
            print(f"  Key Insights: {len(parsed_result.get('key_insights', []))} found")
            
            return parsed_result
        except json.JSONDecodeError:
            print(f"Error parsing JSON for article {article_key}")
            return None
            
    def analyze_all_articles(self):
        """Analyze all articles and compile results"""
        results = {}
        
        for article_key, article_data in self.articles.items():
            analysis = self.analyze_single_article(article_key, article_data)
            if analysis:
                results[article_key] = analysis
                
        if results:
            print(f"\n=== Full Article Analysis Complete ===")
            print(f"Successfully analyzed {len(results)} out of {len(self.articles)} articles")
            
            # Calculate average innovation score
            scores = [r.get('innovation_score', 0) for r in results.values()]
            avg_score = sum(scores) / len(scores) if scores else 0
            print(f"Average Innovation Score: {avg_score:.1f}/10")
            
            # Save results
            self.save_results(results, self.output_filepath)
        else:
            print("No articles were successfully analyzed")
            
    def analyze(self):
        """Required by BaseAnalysis but not used directly"""
        pass


            

if __name__ == "__main__":
    # Step 1: Analyze headlines to find the best ones
    print("=== Step 1: Headline Analysis ===")
    headlines = HeadlineAnalysis()
    
    # Step 2: Get URLs from the best headlines
    urls = headlines.get_urls()
    if not urls:
        print("No URLs found from headline analysis. Exiting.")
        exit()
    
    # Step 3: Scrape full articles
    print("\n=== Step 2: Article Scraping ===")
    news = NewsScraper(urls)
    news.store_news()
    
    # Step 4: Analyze full articles
    print("\n=== Step 3: Full Article Analysis ===")
    full_analysis = FullArticleAnalysis()