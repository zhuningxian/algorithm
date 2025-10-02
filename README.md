## Web Beacon Prototype (Library e-journal pages)

### Overview
This prototype implements a privacy-preserving web beacon for e-journal pages (e.g., CNKI, ScienceDirect):
- Captures anonymous events: page_view, search, click_article, view_abstract, download, time_spent
- Sends to Flask backend: `POST /api/beacon`
- Simple recommendations: `GET /api/recommendations?user_id=...`

### Quickstart
1) Python environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run backend
```bash
export FLASK_APP=beacon_backend/app.py
python beacon_backend/app.py
# or: gunicorn -w 2 -b 0.0.0.0:5000 beacon_backend.app:app
```

3) Verify health
```bash
curl http://localhost:5000/health
```

4) Inject client on target pages
- Tampermonkey: import `tools/tampermonkey.user.js` and visit CNKI/ScienceDirect pages
- Bookmarklet: create a bookmark with content from `tools/bookmarklet.txt` and click on target page

5) Inspect events
```bash
sqlite3 beacon_backend/beacon.db "select event_type, user_id, url, ts_server_iso from events order by id desc limit 20;"
```

6) Get recommendations
```bash
curl "http://localhost:5000/api/recommendations?user_id=<your_user_id>"
```

Notes:
- The script assigns a random `user_id` persisted in localStorage; you can override by setting `window.BEACON_USER_ID` before loading the script.
- To change backend URL, set `window.BEACON_ENDPOINT` or in Tampermonkey set `BEACON_HOST`.

### Event contract (JSON)
```json
{
  "ts": 1710000000000,
  "event_type": "page_view|search|click_article|view_abstract|download|time_spent",
  "user_id": "uuid-v4",
  "session_id": "uuid-v4",
  "site": "sciencedirect.com",
  "url": "https://...",
  "referrer": "https://...",
  "title": "...",
  "search_query": "deep learning",
  "article_id": "/science/article/pii/Sxxx" or "10.1016/j.xxxx",
  "article_title": "...",
  "dwell_ms": 12345,
  "metadata": {"any": "optional"}
}
```

### Security & Privacy
- Do not collect personal identifiers beyond pseudonymous `user_id`.
- Use HTTPS in production; configure CORS to specific host(s).
- Consider rate limiting, API keys, and anonymization for IPs if needed.

### Deploy tips
- Containerize or run via `gunicorn` behind Nginx.
- Persist `beacon_backend/beacon.db` or switch to PostgreSQL.

