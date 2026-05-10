import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"
SITEMAP_URL = "https://www.shl.com/sitemap.xml"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_tags(text: str) -> list[str]:
    chunks = re.split(r"[,/|;]", text)
    return [clean_text(chunk) for chunk in chunks if clean_text(chunk)]


def fetch_url(url: str, timeout: int = 30) -> requests.Response:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(1.5)
    assert last_error is not None
    raise last_error


def parse_detail_page(url: str) -> dict:
    result = {
        "description": "",
        "skills": [],
        "job_roles": [],
        "duration": "",
        "remote_support": "",
        "test_type": "",
        "tags": [],
    }
    try:
        response = fetch_url(url, timeout=30)
        soup = BeautifulSoup(response.text, "html.parser")
    except Exception:
        return result

    # Description extraction from common content blocks.
    desc_node = soup.select_one("meta[name='description']")
    if desc_node and desc_node.get("content"):
        result["description"] = clean_text(desc_node["content"])
    else:
        p = soup.select_one("main p, article p")
        if p:
            result["description"] = clean_text(p.get_text(" ", strip=True))

    text = soup.get_text(" ", strip=True)
    duration_match = re.search(r"(\d+\s*(?:min|minutes))", text, flags=re.IGNORECASE)
    if duration_match:
        result["duration"] = duration_match.group(1)

    if re.search(r"remote|online|virtual", text, flags=re.IGNORECASE):
        result["remote_support"] = "Yes"

    type_match = re.search(r"Test Type\s*:?\s*([A-Za-z0-9 -]+)", text, flags=re.IGNORECASE)
    if type_match:
        result["test_type"] = clean_text(type_match.group(1))

    pill_nodes = soup.select(".tag, .tags a, .category, .pill")
    tags: list[str] = []
    for node in pill_nodes:
        tags.extend(split_tags(node.get_text(" ", strip=True)))
    result["tags"] = sorted(set(tags))

    # Lightweight heuristics for skills and roles based on labels/text snippets.
    skill_match = re.search(r"Skills?\s*:?\s*([^.]*)", text, flags=re.IGNORECASE)
    if skill_match:
        result["skills"] = split_tags(skill_match.group(1))
    role_match = re.search(r"Job Roles?\s*:?\s*([^.]*)", text, flags=re.IGNORECASE)
    if role_match:
        result["job_roles"] = split_tags(role_match.group(1))

    return result


def scrape_catalog() -> list[dict]:
    response = fetch_url(BASE_URL, timeout=30)
    soup = BeautifulSoup(response.text, "html.parser")
    records: list[dict] = []
    cards = soup.select("a[href*='/products/'], .product-catalog a, a[href*='/product-catalog/view/']")
    seen_urls: set[str] = set()

    for card in cards:
        href = card.get("href")
        if not href:
            continue
        url = urljoin("https://www.shl.com", href)
        if url in seen_urls:
            continue
        seen_urls.add(url)

        title = clean_text(card.get_text(" ", strip=True))
        if not title:
            continue
        # Keep only likely individual test items.
        lowered_title = title.lower()
        if "solution" in lowered_title and "individual" not in lowered_title:
            continue

        details = parse_detail_page(url)
        record = {
            "name": title,
            "url": url,
            **details,
        }
        # Restrict to likely individual test pages.
        blob = " ".join([record["name"], record["description"], " ".join(record["tags"])])
        if "individual" in blob.lower() or "test" in blob.lower():
            records.append(record)
    if records:
        return records
    return scrape_catalog_from_sitemap()


def scrape_catalog_from_sitemap() -> list[dict]:
    response = fetch_url(SITEMAP_URL, timeout=30)
    soup = BeautifulSoup(response.text, "xml")
    records: list[dict] = []
    seen_urls: set[str] = set()
    for loc in soup.select("url > loc"):
        url = clean_text(loc.get_text(" ", strip=True))
        if "/products/product-catalog/view/" not in url:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        slug = url.rstrip("/").split("/")[-1].replace("-", " ").strip()
        title = slug.title()
        details = parse_detail_page(url)
        blob = " ".join([title, details.get("description", ""), " ".join(details.get("tags", []))]).lower()
        if "individual" not in blob and "test" not in blob:
            continue
        records.append(
            {
                "name": title,
                "url": url,
                **details,
            }
        )
    return records


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = scrape_catalog()
    OUT_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Saved {len(data)} records to {OUT_PATH}")
