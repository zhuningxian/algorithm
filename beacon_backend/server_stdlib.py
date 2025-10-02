import json
import os
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
DB_PATH = os.path.join(APP_DIR, "beacon.db")
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")


def ensure_dirs() -> None:
    os.makedirs(PUBLIC_DIR, exist_ok=True)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_client_ms INTEGER,
                ts_server_iso TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                site TEXT,
                url TEXT,
                referrer TEXT,
                title TEXT,
                search_query TEXT,
                article_id TEXT,
                article_title TEXT,
                dwell_ms INTEGER,
                ip TEXT,
                ua TEXT,
                metadata_json TEXT
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_user_ts ON events(user_id, ts_client_ms)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events(event_type, ts_client_ms)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_article ON events(article_id)")
        conn.commit()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_event(payload: dict) -> dict:
    allowed_fields = {
        "ts": "ts_client_ms",
        "event_type": "event_type",
        "user_id": "user_id",
        "session_id": "session_id",
        "site": "site",
        "url": "url",
        "referrer": "referrer",
        "title": "title",
        "search_query": "search_query",
        "article_id": "article_id",
        "article_title": "article_title",
        "dwell_ms": "dwell_ms",
        "metadata": "metadata_json",
    }
    sanitized: dict = {}
    for k, dst in allowed_fields.items():
        if k in payload:
            if k == "metadata":
                try:
                    sanitized[dst] = json.dumps(payload[k], ensure_ascii=False)
                except Exception:
                    sanitized[dst] = json.dumps({}, ensure_ascii=False)
            else:
                sanitized[dst] = payload[k]
    return sanitized


def insert_event(event: dict, ip: str, ua: str) -> None:
    record = sanitize_event(event)
    record.setdefault("ts_client_ms", int(time.time() * 1000))
    record.setdefault("dwell_ms", None)
    record.setdefault("user_id", None)
    record.setdefault("session_id", None)
    record.setdefault("site", None)
    record.setdefault("url", None)
    record.setdefault("referrer", None)
    record.setdefault("title", None)
    record.setdefault("search_query", None)
    record.setdefault("article_id", None)
    record.setdefault("article_title", None)
    record.setdefault("metadata_json", None)

    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (
                ts_client_ms, ts_server_iso, event_type, user_id, session_id, site, url, referrer, title,
                search_query, article_id, article_title, dwell_ms, ip, ua, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["ts_client_ms"],
                now_iso(),
                record["event_type"],
                record["user_id"],
                record["session_id"],
                record["site"],
                record["url"],
                record["referrer"],
                record["title"],
                record["search_query"],
                record["article_id"],
                record["article_title"],
                record["dwell_ms"],
                ip,
                ua,
                record["metadata_json"],
            ),
        )
        conn.commit()


def get_recommendations_for_user(user_id: str, limit: int = 10) -> dict:
    with closing(get_db()) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(search_query, '') as q
            FROM events
            WHERE user_id = ? AND (event_type = 'search' OR event_type = 'click_article' OR event_type = 'view_abstract')
            ORDER BY ts_client_ms DESC
            LIMIT 500
            """,
            (user_id,),
        )
        user_queries = [row[0] for row in cur.fetchall() if row[0]]

        def tokenize(q: str) -> list:
            cleaned = (
                q.replace("，", ",")
                .replace("；", ";")
                .replace("|", " ")
                .replace("\u3000", " ")
            )
            parts = []
            for token in cleaned.split():
                token = token.strip(",.;:()[]{}\"'“”‘’！!？?-_")
                if token:
                    parts.append(token.lower())
            return parts

        counts = {}
        for q in user_queries:
            for t in tokenize(q):
                counts[t] = counts.get(t, 0) + 1
        user_top_keywords = [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8]]

        cur.execute(
            """
            SELECT article_id, article_title, url, COUNT(*) as c
            FROM events
            WHERE article_id IS NOT NULL AND article_title IS NOT NULL AND event_type IN ('click_article','view_abstract','download')
                  AND (user_id IS NULL OR user_id <> ?)
            GROUP BY article_id, article_title, url
            ORDER BY c DESC
            LIMIT 200
            """,
            (user_id,),
        )
        candidates = [
            {
                "article_id": r[0],
                "article_title": r[1],
                "url": r[2],
                "count": r[3],
            }
            for r in cur.fetchall()
        ]

        def score(a: dict) -> float:
            title = (a.get("article_title") or "").lower()
            overlap = sum(1 for k in user_top_keywords if k and k in title)
            return float(a.get("count", 0)) + 0.75 * overlap

        ranked = sorted(candidates, key=score, reverse=True)

        if user_top_keywords:
            personalized = [
                a for a in ranked if any(k in (a.get("article_title") or "").lower() for k in user_top_keywords)
            ]
        else:
            personalized = []

        return {
            "user_id": user_id,
            "suggested_keywords": user_top_keywords[:5],
            "personalized_articles": personalized[:limit],
            "trending_articles": ranked[:limit],
        }


def make_json_bytes(obj: dict) -> bytes:
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    server_version = "WBServer/0.1"

    def _set_common_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-User-Id")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_common_headers(204)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/health":
            self._set_common_headers(200)
            self.wfile.write(make_json_bytes({"ok": True}))
            return

        if path == "/beacon.js":
            file_path = os.path.join(PUBLIC_DIR, "beacon.js")
            if not os.path.exists(file_path):
                self._set_common_headers(404)
                self.wfile.write(b"Not found")
                return
            try:
                with open(file_path, "rb") as f:
                    data = f.read()
                self._set_common_headers(200, content_type="text/javascript; charset=utf-8")
                self.wfile.write(data)
            except Exception:
                self._set_common_headers(500)
                self.wfile.write(make_json_bytes({"ok": False, "error": "read_error"}))
            return

        if path == "/api/recommendations":
            query = parse_qs(parsed.query)
            user_id = (query.get("user_id") or [None])[0]
            limit_str = (query.get("limit") or [None])[0]
            if not user_id:
                self._set_common_headers(400)
                self.wfile.write(make_json_bytes({"ok": False, "error": "missing user_id"}))
                return
            try:
                limit = max(1, min(50, int(limit_str))) if limit_str else 10
            except Exception:
                limit = 10
            try:
                data = get_recommendations_for_user(user_id=user_id, limit=limit)
                self._set_common_headers(200)
                self.wfile.write(make_json_bytes({"ok": True, "data": data}))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(make_json_bytes({"ok": False, "error": str(e)}))
            return

        self._set_common_headers(404)
        self.wfile.write(make_json_bytes({"ok": False, "error": "not_found"}))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/beacon":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                self._set_common_headers(400)
                self.wfile.write(make_json_bytes({"ok": False, "error": "invalid_json"}))
                return

            event_type = payload.get("event_type")
            if event_type not in {
                "page_view",
                "search",
                "click_article",
                "view_abstract",
                "download",
                "time_spent",
            }:
                self._set_common_headers(400)
                self.wfile.write(make_json_bytes({"ok": False, "error": "invalid_event_type"}))
                return

            try:
                ip = self.client_address[0] if self.client_address else ""
                ua = self.headers.get("User-Agent", "")
                insert_event(payload, ip, ua)
                self._set_common_headers(200)
                self.wfile.write(make_json_bytes({"ok": True}))
            except Exception as e:
                self._set_common_headers(500)
                self.wfile.write(make_json_bytes({"ok": False, "error": str(e)}))
            return

        self._set_common_headers(404)
        self.wfile.write(make_json_bytes({"ok": False, "error": "not_found"}))


def run(host: str = "0.0.0.0", port: int = 5000) -> None:
    ensure_dirs()
    init_db()
    httpd = HTTPServer((host, port), Handler)
    print(f"[WB] Listening on http://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    run()

