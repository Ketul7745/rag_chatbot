import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import os
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("USER_AGENT", "InsuranceRAGChatbot/1.0")

def is_allowed(url, user_agent=USER_AGENT):
    """Check if URL is allowed by robots.txt."""
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    try:
        response = requests.get(robots_url, headers={"User-Agent": user_agent}, timeout=5)
        rp.parse(response.text.splitlines())
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # Allow if robots.txt is inaccessible

def get_urls_from_page(url, base_domain, support_path, visited, max_urls=500):
    """Extract URLs from a single page, staying within /support."""
    if len(visited) >= max_urls:
        return set()
    
    urls = set()
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(url, href)
            parsed_url = urlparse(absolute_url)
            
            if (parsed_url.netloc == base_domain and
                absolute_url.startswith(f"https://www.angelone.in{support_path}") and
                absolute_url not in visited and
                not parsed_url.fragment):
                if is_allowed(absolute_url):
                    urls.add(absolute_url)
    except Exception as e:
        print(f"Error crawling {url}: {e}")
    
    return urls

def crawl_website(seed_url, support_path="/support", max_urls=500):
    """Crawl all URLs under a website's support section."""
    parsed_seed = urlparse(seed_url)
    base_domain = parsed_seed.netloc
    visited = set()
    to_visit = {seed_url}
    all_urls = set()
    
    while to_visit and len(all_urls) < max_urls:
        current_url = to_visit.pop()
        if current_url in visited:
            continue
            
        print(f"Crawling: {current_url}")
        visited.add(current_url)
        new_urls = get_urls_from_page(current_url, base_domain, support_path, visited, max_urls)
        all_urls.add(current_url)
        to_visit.update(new_urls - visited)
    
    return sorted(list(all_urls))

def save_urls(urls, output_file="crawled_urls.txt"):
    """Save URLs to a file."""
    with open(output_file, "w") as f:
        for url in urls:
            f.write(f"{url}\n")
    print(f"Saved {len(urls)} URLs to {output_file}")

if __name__ == "__main__":
    seed_url = "https://www.angelone.in/support"
    crawled_urls = crawl_website(seed_url, max_urls=500)
    save_urls(crawled_urls)