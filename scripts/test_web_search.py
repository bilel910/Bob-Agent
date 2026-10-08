import json
import httpx
from ai_companion.settings import settings

response = httpx.post(
    "https://api.tavily.com/search",
    headers={"Authorization": f"Bearer {settings.TAVILY_API_KEY}"},
    json={"query": "Hertha BSC Ergebnis", "topic": "news",
          "max_results": 3, "include_answer": True},
    timeout=20,
)

print(response.status_code)
print(json.dumps(response.json(), indent=2, ensure_ascii=False)[:3000])
