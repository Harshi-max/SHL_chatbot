import json

import requests


def run_sample(base_url: str = "http://localhost:8000") -> dict:
    payload = {
        "messages": [
            {"role": "user", "content": "Hiring a Java developer with client-facing responsibilities"},
            {"role": "assistant", "content": "Do you need personality assessment too?"},
            {"role": "user", "content": "Yes include personality tests."},
        ]
    }
    response = requests.post(f"{base_url}/chat", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    output = run_sample()
    print(json.dumps(output, indent=2))
