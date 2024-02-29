import re
from collections import Counter
from typing import Dict

import requests


def chech_availability_for_message(urls: list):
    reachable_urls = Counter()
    http_patterns = re.compile(r'^https?://')
    for url in urls:
        try:
            url = re.sub(r'#.*$', '', url)
            url = re.sub(r'https?://', '', url)
            url = re.sub(r'www\.', '', url)
            response = requests.get(f"https://{url}", timeout=5)
            reachable_urls.update([url])

        except requests.RequestException as e:
            print(e)
    return dict(reachable_urls)



def parse_urls(message: str) -> Dict[str, int]:
    url_pattern = re.compile(r'\b(?:https?://)?(?:www\.)?(?:xn--)?\S+\.\S{2,}\b')
    urls = url_pattern.findall(message)
    result = chech_availability_for_message(urls)
    return result


if __name__ == "__main__":
    message = (
        "Long string with multiple URLs: Check out the news at https: // www.nytimes.com, grab some tech gadgets from https: // www.bestbuy.com, find home improvement supplies at https: // www.homedepot.com, look for health information on https: // www.webmd.com, learn coding at https: // www.codecademy.com, get financial advice from https: // www.forbes.com, and explore https: // en.wikipedia.org / wiki / Special:Random for a random Wikipedia article."
    )
    print(parse_urls(message))
