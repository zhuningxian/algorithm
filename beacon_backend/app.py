import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil import tz
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


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
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_article ON events(article_id)"
        )
        conn.commit()


def now_iso() -> str:
    return (
        datetime.now(tz=tz.tzlocal())
        .astimezone(tz.tzutc())
        .isoformat()
        .replace("+00:00", "Z")
    )


def sanitize_event(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    sanitized: Dict[str, Any] = {}
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


def insert_event(event: Dict[str, Any], ip: str, ua: str) -> None:
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


def get_client_ip() -> str:
    fwd = request.headers.get("X-Forwarded-For")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.remote_addr or ""


def get_recommendations_for_user(user_id: str, limit: int = 10) -> Dict[str, Any]:
    # Simple heuristic: extract user top keywords, then rank articles from other users by keyword overlap and popularity.
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

        def tokenize(q: str) -> List[str]:
            cleaned = q.replace("，", ",").replace("；", ";").replace("|", " ")
            parts = []
            for token in cleaned.replace("\u3000", " ").split():
                token = token.strip().strip(",.;:()[]{}\"'“”‘’！!？?-_")
                if token:
                    parts.append(token.lower())
            return parts

        keyword_counter: Counter[str] = Counter()
        for q in user_queries:
            keyword_counter.update(tokenize(q))
        user_top_keywords = [w for w, _ in keyword_counter.most_common(8)]

        # Candidate popular articles from other users
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
        candidates = [dict(article_id=r[0], article_title=r[1], url=r[2], count=r[3]) for r in cur.fetchall()]

        def score_article(a: Dict[str, Any]) -> float:
            title = (a.get("article_title") or "").lower()
            overlap = sum(1 for k in user_top_keywords if k and k in title)
            return a.get("count", 0) + 0.75 * overlap

        ranked = sorted(candidates, key=score_article, reverse=True)

        # Trending if personalized empty
        trending = ranked[: limit if ranked else 10]

        # Personalized filter: require some overlap if we have keywords
        if user_top_keywords:
            personalized = [a for a in ranked if any(k in (a.get("article_title") or "").lower() for k in user_top_keywords)]
        else:
            personalized = []

        return {
            "user_id": user_id,
            "suggested_keywords": user_top_keywords[:5],
            "personalized_articles": personalized[:limit],
            "trending_articles": trending[:limit],
        }


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)


@app.before_first_request
def _boot() -> None:
    ensure_dirs()
    init_db()


@app.get("/health")
def health():
    return jsonify({"ok": True})


@app.get("/beacon.js")
def serve_beacon_js():
    # Serve client beacon script for easy injection
    if not os.path.exists(os.path.join(PUBLIC_DIR, "beacon.js")):
        return ("Not found", 404)
    return send_from_directory(PUBLIC_DIR, "beacon.js", mimetype="text/javascript")


@app.post("/api/beacon")
def api_beacon():
    try:
        payload = request.get_json(force=True, silent=False) or {}
    except Exception:
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    event_type = payload.get("event_type")
    if event_type not in {
        "page_view",
        "search",
        "click_article",
        "view_abstract",
        "download",
        "time_spent",
    }:
        return jsonify({"ok": False, "error": "invalid_event_type"}), 400

    try:
        insert_event(payload, get_client_ip(), request.headers.get("User-Agent", ""))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    return jsonify({"ok": True})


@app.get("/api/recommendations")
def api_recommendations():
    user_id = request.args.get("user_id") or request.headers.get("X-User-Id")
    if not user_id:
        return jsonify({"ok": False, "error": "missing user_id"}), 400
    limit = 10
    try:
        if request.args.get("limit"):
            limit = max(1, min(50, int(request.args["limit"])))
    except Exception:
        pass

    rec = get_recommendations_for_user(user_id=user_id, limit=limit)
    return jsonify({"ok": True, "data": rec})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

