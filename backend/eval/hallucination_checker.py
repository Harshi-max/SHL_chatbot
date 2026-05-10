import json
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "catalog.json"


def check_hallucinations(recommendation_payload: dict) -> list[str]:
    catalog = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    catalog_names = {item["name"] for item in catalog}
    catalog_urls = {item["url"] for item in catalog}
    issues: list[str] = []
    for rec in recommendation_payload.get("recommendations", []):
        if rec.get("name") not in catalog_names:
            issues.append(f"Unknown assessment name: {rec.get('name')}")
        if rec.get("url") not in catalog_urls:
            issues.append(f"Unknown assessment url: {rec.get('url')}")
    return issues
