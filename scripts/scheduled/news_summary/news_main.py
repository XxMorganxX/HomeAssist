import argparse
import os
from news_scraper import NewsHeadlineScraper, NewsScraper
from news_ai import HeadlineAnalysis, FullArticleAnalysis, NewsSummaryGenerator


def clear_ephemeral_data():
    """Remove all files in ephemeral_data folder except .gitkeep"""
    ephemeral_dir = os.path.join(os.path.dirname(__file__), "ephemeral_data")
    
    if not os.path.exists(ephemeral_dir):
        print(f"Directory not found: {ephemeral_dir}")
        return
    
    removed_count = 0
    for filename in os.listdir(ephemeral_dir):
        if filename == ".gitkeep":
            continue
        filepath = os.path.join(ephemeral_dir, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Removed: {filename}")
            removed_count += 1
    
    print(f"\nCleared {removed_count} file(s) from ephemeral_data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News scraping and analysis pipeline")
    parser.add_argument("--clean", action="store_true", help="Clear all files in ephemeral_data/ except .gitkeep")
    args = parser.parse_args()
    
    if args.clean:
        clear_ephemeral_data()
        print()  # Add spacing before starting pipeline

# Step 1: Scrape headlines
    print("=== Step 1: Headline Scraping ===")
    news_fetcher = NewsHeadlineScraper()
    news_fetcher.fetch_headlines()
    news_fetcher.get_num_results_found()
    
    # Step 2: Import and run analysis pipeline
    try:
        from news_ai import HeadlineAnalysis, FullArticleAnalysis
        
        # Step 2a: Analyze headlines to find the best ones
        print("\n=== Step 2: Headline Analysis ===")
        headlines = HeadlineAnalysis()
        
        # Step 2b: Get URLs from the best headlines
        urls = headlines.get_urls()
        if not urls:
            print("No URLs found from headline analysis. Exiting.")
            exit()
        
        # Step 2c: Scrape full articles
        print("\n=== Step 3: Article Scraping ===")
        news = NewsScraper(urls)
        news.store_news()
        
        # Step 2d: Analyze full articles
        print("\n=== Step 4: Full Article Analysis ===")
        full_analysis = FullArticleAnalysis()
        
        # Step 2e: Generate concise summary
        print("\n=== Step 5: News Summary Generation ===")
        summary_generator = NewsSummaryGenerator()
        
        print("\n🎉 Complete analysis pipeline finished!")
        print("Check the following files for results:")
        print("  - ephemeral_data/headlines.json (scraped headlines)")
        print("  - ephemeral_data/top_headlines.json (best headlines)")
        print("  - ephemeral_data/news.json (article content)")
        print("  - ephemeral_data/full_article_analysis.json (detailed analysis)")
        print("  - ephemeral_data/news_summary.json (concise summary)")
        
    except ImportError as e:
        print(f"\nAnalysis modules not available: {e}")
        print("Headlines have been scraped and saved to ephemeral_data/headlines.json")
    except Exception as e:
        print(f"\nError during analysis: {type(e).__name__}: {str(e)}")
        print("Headlines have been scraped and saved to ephemeral_data/headlines.json")