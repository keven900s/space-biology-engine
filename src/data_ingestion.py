import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pickle
from pathlib import Path

class DataIngestion:
    def __init__(self, config):
        self.config = config
        self.papers = []
        
    def load_csv(self):
        """Load the CSV with paper titles and PMC links"""
        df = pd.read_csv(self.config.CSV_FILE)
        print(f"Loaded {len(df)} papers from CSV")
        return df
    
    def scrape_paper_content(self, pmc_url, max_retries=3):
        """Scrape full text from PubMed Central"""
        for attempt in range(max_retries):
            try:
                headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/113.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}
                response = requests.get(pmc_url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Extract sections
                abstract = self._extract_section(soup, ['abstract', 'Abstract'])
                introduction = self._extract_section(soup, ['introduction', 'Introduction'])
                methods = self._extract_section(soup, ['methods', 'Methods', 'materials'])
                results = self._extract_section(soup, ['results', 'Results'])
                discussion = self._extract_section(soup, ['discussion', 'Discussion'])
                conclusion = self._extract_section(soup, ['conclusion', 'Conclusion', 'conclusions'])
                
                return {
                    'abstract': abstract,
                    'introduction': introduction,
                    'methods': methods,
                    'results': results,
                    'discussion': discussion,
                    'conclusion': conclusion,
                    'full_text': soup.get_text(separator=' ', strip=True)[:10000]  # First 10k chars
                }
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to scrape {pmc_url}: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def _extract_section(self, soup, section_keywords):
        """Helper to extract specific sections"""
        for keyword in section_keywords:
            section = soup.find(['h2', 'h3', 'div'], string=lambda text: text and keyword.lower() in text.lower())
            if section:
                # Get all text until next heading
                text = []
                for sibling in section.find_next_siblings():
                    if sibling.name in ['h2', 'h3']:
                        break
                    text.append(sibling.get_text(strip=True))
                return ' '.join(text)
        return ""
    
    def process_all_papers(self, df, use_cache=True):
        """Process all papers with caching"""
        cache_file = self.config.PROCESSED_DIR / "papers_cache.pkl"
        
        if use_cache and cache_file.exists():
            print("Loading from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        papers = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scraping papers"):
            paper = {
                'id': idx,
                'title': row.iloc[0] if len(row) > 0 else "",
                'pmc_url': row.iloc[1] if len(row) > 1 else "",
            }
            
            # Scrape content (with rate limiting)
            if paper['pmc_url']:
                content = self.scrape_paper_content(paper['pmc_url'])
                if content:
                    paper.update(content)
                time.sleep(0.5)  # Be nice to PMC servers
            
            papers.append(paper)
            
            # Save checkpoint every 50 papers
            if (idx + 1) % 50 == 0:
                with open(cache_file, 'wb') as f:
                    pickle.dump(papers, f)
        
        # Final save
        with open(cache_file, 'wb') as f:
            pickle.dump(papers, f)
        
        return papers
