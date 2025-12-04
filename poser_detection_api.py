from __future__ import annotations
import os
import re
import time
import threading
import requests
import hashlib
import json
from dotenv import load_dotenv
from apify_client import ApifyClient
from dateutil import parser as date_parser
import firebase_admin
from firebase_admin import credentials, firestore
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. AI AGENT SETUP (Groq) ---
try:
    from groq import Groq
except ImportError:
    print("Warning: 'groq' library not installed. AI Agent disabled.")
    Groq = None

app = Flask(__name__)
_allowed_raw = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed = [o.strip() for o in _allowed_raw.split(",") if o.strip()] or ["*"]
try:
    CORS(app, resources={r"/api/*": {"origins": _allowed}}, supports_credentials=False)
except Exception:
    CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

@app.route('/api/health')
def _health():
    return jsonify({"status": "ok"})

def _load_env_var(key: str, default: str = "") -> str:
    v = os.getenv(key)
    if v: return v
    try:
        env_path = os.path.join(os.path.dirname(__file__), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or '=' not in line: continue
                    k, val = line.split("=", 1)
                    if k.strip() == key: return val.strip().strip('"').strip("'").strip('`')
    except Exception: pass
    return default

load_dotenv()

# --- CONFIGURATION ---
APIFY_TOKEN = (os.getenv("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN") or _load_env_var("APIFY_TOKEN"))
META_GRAPH_TOKEN = _load_env_var("META_GRAPH_TOKEN")
GRAPH_BASE_URL = (_load_env_var("GRAPH_BASE_URL", "https://graph.facebook.com/v24.0") or "https://graph.facebook.com/v24.0").strip().strip('`')
META_APP_ID = _load_env_var("META_APP_ID")
META_APP_SECRET = _load_env_var("META_APP_SECRET")
POSER_ADMIN_SECRET = _load_env_var("POSER_ADMIN_SECRET")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POSER_LOG_COLLECTION = _load_env_var("POSER_LOG_COLLECTION", "poser_detections")
FORCE_APIFY = str(_load_env_var("FORCE_APIFY", "False")).strip().lower() in ("true","1","yes","y")

REQUIRED_SCOPES = ["pages_show_list", "pages_read_engagement", "pages_read_user_content"] 

GRAPH_CACHE_TTL = int(_load_env_var("GRAPH_CACHE_TTL", "300"))
GRAPH_CACHE_MAX_ENTRIES = int(_load_env_var("GRAPH_CACHE_MAX_ENTRIES", "500"))
_GRAPH_CACHE_LOCK = threading.Lock()
GRAPH_CACHE: Dict[str, Dict[str, Any]] = {}
LAST_GRAPH_ERROR: Optional[Dict[str, Any]] = None

# --- DATABASE COLLECTIONS ---
RAW_DATA_COLLECTION = "analyzed_pages_cache" 
VERDICT_COLLECTION = "poser_detections" 
REGISTRY_COLLECTION = "verified_registry"
OLD_CACHE_COLLECTION = "analyzed_pages_cache" 

# Initialize AI Client
AI_AGENT_REASON = None
groq_lib_present = bool(Groq)
groq_key_present = bool(GROQ_API_KEY)
if groq_lib_present and groq_key_present:
    groq_client = Groq(api_key=GROQ_API_KEY)
    AI_AGENT_REASON = "ok"
    print("AI Agent Online (Llama 3 via Groq)")
else:
    groq_client = None
    AI_AGENT_REASON = "missing_library" if not groq_lib_present else "missing_key"
    print("AI Agent Offline (Missing Key or Library)")

if not firebase_admin._apps:
    try:
        # 1. Get the entire JSON file contents as a string from the environment
        service_account_json_string = os.environ.get('FIREBASE_CONFIG_JSON')
        
        if service_account_json_string:
            # Step 1a: AGGRESSIVE STRIPPING (Cleans hidden characters/quotes)
            clean_json_string = service_account_json_string.strip().strip('"').strip("'")

            # Step 1b: CRITICAL FIX: Replace escaped newlines with actual newlines
            fixed_json_string = clean_json_string.replace('\\n', '\n') 
            
            # 2. Convert the fixed string back into a Python dictionary/JSON object
            service_account_info = json.loads(fixed_json_string)
            
            # 3. Initialize the credentials using the dictionary
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
            print("Connected to CrediNews Firebase via Environment Variable!")
        else:
            # Fallback for local development
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
            print("Connected to CrediNews Firebase via Local File!")

    except Exception as e:
        print(f"Firebase Connection Error: {e}")

try:
    db = firestore.client()
except Exception:
    db = None
    print(" Firestore client could not be initialized.")

def _get_cache_key(url: str) -> str:
    return hashlib.md5(url.strip().lower().encode('utf-8')).hexdigest()

def _require_admin_secret(data: Optional[Dict[str, Any]] = None) -> bool:
    secret = (POSER_ADMIN_SECRET or "").strip()
    if not secret: return False

    if request.headers.get("X-Admin-Secret") == secret:
        return True

    if data and data.get("admin_secret") == secret:
        return True
        
    return False

def _make_cache_key(path: str, params: Optional[Dict[str, Any]]) -> str:
    try:
        items = sorted(((params or {}).items()))
        return f"{path.strip('/')}?" + "&".join([f"{k}={v}" for k, v in items])
    except Exception: return path.strip("/")

def _check_verified_registry(url: str) -> Optional[Dict[str, Any]]:
    try:
        if not db: return None
        raw = (url or "").strip()
        clean = raw.strip('`').strip('"').strip("'")
        candidates = [clean]
        try:
            from urllib.parse import urlparse
            u = urlparse(clean)
            scheme = (u.scheme or 'https').lower()
            host = (u.netloc or '').lower()
            path = (u.path or '/').strip() or '/'
            if not path.startswith('/'): path = '/' + path
            base_no_slash = f"{scheme}://{host}{path.rstrip('/')}"
            base_with_slash = f"{scheme}://{host}{(path.rstrip('/') + '/') if path != '/' else '/'}"
            candidates.extend([base_no_slash, base_with_slash])
            if host.startswith('www.'):
                host2 = host[4:]
                candidates.extend([
                    f"{scheme}://{host2}{path}",
                    f"{scheme}://{host2}{path.rstrip('/')}",
                    f"{scheme}://{host2}{(path.rstrip('/') + '/') if path != '/' else '/'}"
                ])
            else:
                host2 = 'www.' + host if host else host
                if host2:
                    candidates.extend([
                        f"{scheme}://{host2}{path}",
                        f"{scheme}://{host2}{path.rstrip('/')}",
                        f"{scheme}://{host2}{(path.rstrip('/') + '/') if path != '/' else '/'}"
                    ])
                if host:
                    mhost = ('m.' + host) if not host.startswith('m.') else host
                    candidates.extend([
                        f"{scheme}://{mhost}{path}",
                        f"{scheme}://{mhost}{path.rstrip('/')}",
                        f"{scheme}://{mhost}{(path.rstrip('/') + '/') if path != '/' else '/'}"
                    ])
        except Exception:
            pass
        try:
            candidates.append(_normalize_to_page_url(clean))
        except Exception:
            pass

        for u in candidates:
            try:
                did = _get_cache_key(u)
                snap = db.collection("verified_registry").document(did).get()
                if snap.exists:
                    return snap.to_dict()
            except Exception:
                continue

        for u in candidates:
            try:
                qs = db.collection("verified_registry").where("url", "==", u).limit(1).get()
                for s in qs:
                    return s.to_dict()
            except Exception:
                continue
    except Exception:
        pass
    return None

def _set_last_graph_error(status_code: Any, details: str) -> None:
    global LAST_GRAPH_ERROR
    try:
        LAST_GRAPH_ERROR = {
            "at": datetime.now(timezone.utc).isoformat(),
            "status_code": status_code,
            "details": (details or "")[:500]
        }
    except Exception: pass

def _graph_get(path: str, params: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> Dict[str, Any]:
    if not META_GRAPH_TOKEN: return {"error": "Missing META_GRAPH_TOKEN"}
    params = dict(params or {})
    params["access_token"] = META_GRAPH_TOKEN
    url = f"{GRAPH_BASE_URL}/{path.strip('/')}"
    cache_key = _make_cache_key(path, params)
    now_ts = time.time()
    if use_cache:
        with _GRAPH_CACHE_LOCK:
            cached = GRAPH_CACHE.get(cache_key)
            if cached and cached.get("expires_at", 0) > now_ts: return cached["data"]
    attempts = 0
    backoff = 0.5
    last_error = None
    while attempts < 3:
        attempts += 1
        try:
            resp = requests.get(url, params=params, timeout=10)
            try: body = resp.json()
            except Exception: body = None
            if resp.status_code == 200:
                data = body if isinstance(body, dict) else body or {}
                if use_cache:
                    with _GRAPH_CACHE_LOCK:
                        GRAPH_CACHE[cache_key] = {"data": data, "expires_at": now_ts + GRAPH_CACHE_TTL}
                return data
            if isinstance(body, dict) and "error" in body:
                err = body.get("error") or {}
                if err.get("code") == 190:
                    _set_last_graph_error(resp.status_code, str(err))
                    return {"error": "OAuthException", "details": err}
            if resp.status_code in (429, 500, 502, 503):
                time.sleep(backoff)
                backoff *= 2
                continue
            _set_last_graph_error(resp.status_code, str(body or resp.text))
            return {"error": f"Graph error {resp.status_code}", "details": body or resp.text}
        except Exception as e:
            last_error = str(e)
            time.sleep(backoff)
            backoff *= 2
            continue
    _set_last_graph_error(None, str(last_error))
    return {"error": "Graph request failed after retries", "details": last_error}

def _has_graph_error(obj: Any) -> bool:
    return isinstance(obj, dict) and bool(obj.get("error"))

def _debug_token_info(access_token: str) -> Dict[str, Any]:
    if not access_token or not META_APP_ID or not META_APP_SECRET: return {}
    try:
        app_token = f"{META_APP_ID}|{META_APP_SECRET}"
        resp = requests.get(f"{GRAPH_BASE_URL}/debug_token", params={"input_token": access_token, "access_token": app_token}, timeout=10)
        data = resp.json().get("data") if resp.status_code == 200 else {}
        return {"is_valid": bool(data.get("is_valid"))}
    except Exception: return {}

def _apify_health() -> Dict[str, Any]:
    try:
        if not APIFY_TOKEN: return {"token_loaded": False}
        client = ApifyClient(APIFY_TOKEN)
        u = client.user().get()
        return {"token_loaded": True, "user_id": (u or {}).get("id")}
    except Exception as e: return {"token_loaded": bool(APIFY_TOKEN), "error": str(e)[:200]}

def parse_url(url: str) -> Dict[str, Any]:
    try:
        from urllib.parse import urlparse
        u = urlparse((url or "").strip())
        host = (u.netloc or "").lower()
        if not host: return {"is_valid": False, "error": "Missing host"}
        if "facebook.com" not in host and "fb.com" not in host: return {"is_valid": False, "error": "Invalid URL"}
        return {"is_valid": True, "normalized_url": url}
    except Exception: return {"is_valid": False, "error": "Invalid URL"}

def extract_fbid(url_or_id: str) -> str:
    if not url_or_id: return ""
    s = url_or_id.strip()
    if re.match(r"^[A-Za-z0-9_.-]+$", s): return s
    try:
        from urllib.parse import urlparse, parse_qs
        u = urlparse(s)
        qs = parse_qs(u.query or "")
        if "id" in qs and qs["id"]: return qs["id"][0]
        parts = [p for p in (u.path or "").split("/") if p]
        for p in parts:
            if re.match(r"^\d{5,}$", p): return p
        if parts: return parts[0]
    except Exception: pass
    return s

def _extract_first_link(text: str) -> Optional[str]:
    """Helper to find the first http/https link in a text block"""
    if not text: return None
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(1) if match else None

def _resolve_fb_share_url(share_url: str) -> Optional[str]:
    try:
        if not share_url:
            return None
        r = requests.get(share_url, timeout=10, headers={"User-Agent": "CrediNews-Bot/1.0"})
        if r.status_code != 200:
            return None
        html = r.text or ""
        m = re.search(r'<meta\s+property=["\']og:url["\']\s+content=["\'](https?://[^"\']+)["\']', html, re.IGNORECASE)
        if m:
            return m.group(1)
        m2 = re.search(r'"permalink_url"\s*:\s*"(https?://[^"]+)"', html, re.IGNORECASE)
        if m2:
            return m2.group(1)
        m3 = re.search(r'"owner_id"\s*:\s*"?(\d+)"?', html, re.IGNORECASE)
        if m3:
            oid = m3.group(1)
            return f"https://www.facebook.com/{oid}"
        return None
    except Exception:
        return None

def _normalize_to_page_url(u: str) -> str:
    try:
        from urllib.parse import urlparse
        url = urlparse(u)
        path = (url.path or '').lower()
        stops = ['/posts/', '/videos/', '/photos/', '/reel/', '/story.php', '/permalink.php']
        for stop in stops:
            idx = path.find(stop)
            if idx > 1:
                base = (url.scheme or 'https') + '://' + (url.netloc or '') + (url.path[:idx] or '/')
                return base
        return (url.scheme or 'https') + '://' + (url.netloc or '') + (url.path or '/')
    except Exception:
        return u

def fetch_metadata(fbid: str) -> Dict[str, Any]:
    base_fields = ["name", "link", "created_time", "start_info", "founded", "birthday"]
    res = _graph_get(fbid, {"fields": ",".join(base_fields)}, use_cache=True)
    
    if _has_graph_error(res): return res
    if not isinstance(res, dict): res = {}

    final_created_time = res.get("created_time")
    if not final_created_time:
        start = res.get("start_info", {})
        if isinstance(start, dict) and start.get("date"):
            try: 
                final_created_time = date_parser.parse(start.get("date")).isoformat()
            except: pass

    if not final_created_time and res.get("founded"):
        try: 
            final_created_time = date_parser.parse(str(res.get("founded"))).isoformat()
        except: pass

    if final_created_time:
        res["created_time"] = final_created_time

    optional_fields = ["category","about","description","followers_count","website","verification_status","is_verified"]
    restricted = False
    for fld in optional_fields:
        r = _graph_get(fbid, {"fields": fld}, use_cache=True)
        if _has_graph_error(r):
            try:
                if int(((r.get("details") or {}).get("error") or {}).get("code") or 0) == 10: restricted = True
            except: pass
        elif isinstance(r, dict) and fld in r: res[fld] = r.get(fld)
    
    pic_r = _graph_get(fbid, {"fields": "picture{url,is_silhouette}"}, use_cache=True)
    if not _has_graph_error(pic_r) and isinstance(pic_r, dict) and pic_r.get("picture"): res["picture"] = pic_r.get("picture")
    
    cover_r = _graph_get(fbid, {"fields": "cover{source}"}, use_cache=True)
    if not _has_graph_error(cover_r) and isinstance(cover_r, dict) and cover_r.get("cover"): res["cover"] = cover_r.get("cover")
    
    posts = _graph_get(f"{fbid}/posts", {"limit": 10, "fields": "created_time"}, use_cache=True)
    if not _has_graph_error(posts) and isinstance(posts.get("data"), list):
        res["recent_posts_count"] = len(posts.get("data") or [])
        try:
            times = []
            for p in (posts.get("data") or [])[:5]:
                ct = p.get("created_time")
                if ct:
                    dt = datetime.fromisoformat(str(ct).replace("Z", "+00:00"))
                    times.append(dt)
            if len(times) >= 2:
                newest = max(times)
                oldest = min(times)
                diff = newest - oldest
                seconds = int(diff.total_seconds())
                if seconds < 60:
                    res["post_time_span"] = f"in {seconds} seconds"
                elif seconds < 3600:
                    res["post_time_span"] = f"in {seconds // 60} minutes"
                elif seconds < 86400:
                    res["post_time_span"] = f"in {seconds // 3600} hours"
                else:
                    res["post_time_span"] = f"across {seconds // 86400} days"
        except Exception:
            pass
        
    res["resource_type"] = "page" if (res.get("fan_count") is not None) else "profile"
    res["_permissions_restricted"] = restricted
    res["_source"] = "graph" 
    return res

def _parse_apify_date(s: Optional[str]) -> Optional[str]:
    try:
        if not s: return None
        dt = date_parser.parse(str(s))
        if not dt.tzinfo: dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception: return None

def run_apify_scraper(page_url: str) -> Optional[Dict[str, Any]]:
    if db:
        doc_id = _get_cache_key(page_url)
        try:
            doc = db.collection(RAW_DATA_COLLECTION).document(doc_id).get()
            if doc.exists:
                cached = doc.to_dict() or {}
                has_sufficient = bool(cached.get("name")) and (
                    bool(cached.get("recent_posts_count")) or bool(((cached.get("picture") or {}).get("data") or {}).get("url")) or bool((cached.get("cover") or {}).get("source"))
                )
                if has_sufficient:
                    print(f"Found in Raw Cache (skipping Apify): {page_url}")
                    return cached
        except Exception as e:
            print(f"Firebase Read Warning: {e}")

    try:
        token = APIFY_TOKEN
        if not token: return None
        client = ApifyClient(token)
        print(f"Starting Apify Scan for: {page_url}...")

        run_page = client.actor("apify/facebook-pages-scraper").call(run_input={"startUrls": [{"url": page_url}]})
        page_data = {}
        if run_page:
            for item in client.dataset(run_page["defaultDatasetId"]).iterate_items():
                page_data = item
                break

        run_posts = client.actor("apify/facebook-posts-scraper").call(
            run_input={"startUrls": [{"url": page_url}], "resultsLimit": 5, "viewPort": {"width": 1920, "height": 1080}}
        )
        raw_posts = []
        if run_posts:
            for item in client.dataset(run_posts["defaultDatasetId"]).iterate_items():
                raw_posts.append(item)

        spam_score = 0
        post_time_span = "" 
        
        repeated_link_penalty = 0
        if len(raw_posts) >= 3:
            last_links = []
            for p in raw_posts[:3]:
                link = p.get("link") or p.get("url") 
                if not link:
                    txt = p.get("text") or p.get("postText") or ""
                    link = _extract_first_link(txt)
                if link:
                    last_links.append(link.strip().lower())
            
            if len(last_links) == 3 and (last_links[0] == last_links[1] == last_links[2]):    
                print(f"DETECTED REPEATED LINKS: {last_links[0]}")
                repeated_link_penalty = -20
        
        if len(raw_posts) >= 2:
            try:
                timestamps = []
                for p in raw_posts:
                    ts = p.get("time") or p.get("timestamp")
                    if ts: timestamps.append(date_parser.parse(str(ts)))
                
                if len(timestamps) >= 2:
                    newest = max(timestamps)
                    oldest = min(timestamps)
                    diff = newest - oldest
                    total_seconds = diff.total_seconds()
                    
                    if len(timestamps) >= 5:
                        if total_seconds < 3600: spam_score = -20
                        if total_seconds < 600: spam_score = -40
                    
                    if total_seconds < 60:
                        post_time_span = f"in {int(total_seconds)} seconds"
                    elif total_seconds < 3600:
                        post_time_span = f"in {int(total_seconds // 60)} minutes"
                    elif total_seconds < 86400:
                        post_time_span = f"in {int(total_seconds // 3600)} hours"
                    else:
                        days = int(total_seconds // 86400)
                        post_time_span = f"across {days} days"
            except: pass

        final_spam_score = spam_score + repeated_link_penalty

        created_iso = _parse_apify_date(page_data.get("pageCreationDate"))
        pic_url = page_data.get("profilePicture")
        if not pic_url and raw_posts: pic_url = raw_posts[0].get("user", {}).get("profilePic")
        name_found = page_data.get("name") or page_data.get("title")
        if not name_found and raw_posts: name_found = raw_posts[0].get("user", {}).get("name")

        apify_verified = bool(page_data.get("verified")) or bool(page_data.get("isVerified")) or bool(page_data.get("is_meta_verified"))
        if not apify_verified:
            try:
                for p in raw_posts:
                    u = p.get("user") or {}
                    if bool(u.get("isVerified")) or bool(u.get("verified")):
                        apify_verified = True
                        break
            except Exception:
                pass

        badge_name = str(page_data.get("badge") or "").strip().lower()
        if badge_name == "blue":
            apify_verified = True

        def _safe_int(val: Any) -> int:
            try:
                if val is None:
                    return 0
                if isinstance(val, (int, float)):
                    return int(val)
                s = str(val).strip().replace(',', '')
                if s[-1:].upper() in ('K','M'):
                    mult = 1000 if s[-1:].upper() == 'K' else 1000000
                    num = float(s[:-1]) if s[:-1] else 0.0
                    return int(num * mult)
                return int(float(s))
            except Exception:
                return 0

        # Ensure username exists
        extracted_username = page_data.get("username")
        if not extracted_username:
            extracted_username = extract_fbid(page_url)

        meta = {
            "id": page_data.get("id") or page_data.get("facebookId"),
            "name": name_found,
            "username": extracted_username,
            "fan_count": _safe_int(page_data.get("likes")),
            "followers_count": _safe_int(page_data.get("followers")),
            "created_time": created_iso,
            "is_verified": apify_verified,
            "verification_status": "blue_verified" if apify_verified else "not_verified",
            "link": page_url,
            "website": page_data.get("website"),
            "picture": { "data": { "url": pic_url, "is_silhouette": not bool(pic_url) } },
            "cover": { "source": page_data.get("coverPhotoUrl") },
            "about": page_data.get("intro") or page_data.get("bio") or "",
            "recent_posts_count": len(raw_posts), 
            "spam_score": final_spam_score,
            "post_time_span": post_time_span,
            "resource_type": "page",
            "_apify_fallback_used": True
        }

        meta["raw_post_texts"] = [p.get("text") or p.get("postText") or "" for p in raw_posts[:3]]

        try:
            if (meta["followers_count"] or 0) == 0 or (meta["fan_count"] or 0) == 0:
                for p in raw_posts:
                    u = p.get("user") or {}
                    fc = _safe_int(u.get("followers") or u.get("followersCount") or u.get("fans") or u.get("fanCount"))
                    lk = _safe_int(u.get("likes") or u.get("likeCount"))
                    if (meta["followers_count"] or 0) == 0 and fc > 0:
                        meta["followers_count"] = fc
                    if (meta["fan_count"] or 0) == 0 and lk > 0:
                        meta["fan_count"] = lk
                    if meta["followers_count"] > 0 and meta["fan_count"] > 0:
                        break
        except Exception:
            pass

        if not meta.get("name"): return None
        
        if db:
            try:
                doc_id = _get_cache_key(page_url)
                meta["_cached_at"] = datetime.now(timezone.utc).isoformat()
                meta["_original_url"] = page_url
                db.collection(RAW_DATA_COLLECTION).document(doc_id).set(meta, merge=True)
            except Exception: pass

        return meta
    except Exception: return None

def _merge_apify_into_meta(graph_meta: Dict[str, Any], apify_meta: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(graph_meta or {})
    if not apify_meta: return merged
    
    for k in ["fan_count","followers_count","created_time","website","link","name","username","about","description","recent_posts_count","spam_score", "post_time_span"]:
        val = apify_meta.get(k)
        if val not in (None, "", 0, [], {}):
            if merged.get(k) in (None, "", 0, [], {}):
                merged[k] = val
    
    if apify_meta.get("raw_post_texts"):
        merged["raw_post_texts"] = apify_meta.get("raw_post_texts")
    
    status_g = str(merged.get("verification_status") or "").strip().lower()
    graph_has_badge = bool(merged.get("is_verified")) or status_g in ("blue_verified", "verified", "meta_verified")
    status_a = str(apify_meta.get("verification_status") or "").strip().lower()
    apify_has_badge = bool(apify_meta.get("is_verified")) or bool(apify_meta.get("verified")) or status_a in ("blue_verified", "verified", "meta_verified") or str(apify_meta.get("badge") or "").strip().lower() == "blue" or bool(apify_meta.get("isVerified"))

    if graph_has_badge or apify_has_badge:
        merged["is_verified"] = True
        merged["verification_status"] = "blue_verified"
    else:
        if not merged.get("is_verified"):
            merged["is_verified"] = False
            merged["verification_status"] = "not_verified"

    apify_pic = apify_meta.get("picture", {}).get("data", {})
    graph_pic = merged.get("picture", {}).get("data", {})
    
    if apify_pic.get("url") and not apify_pic.get("is_silhouette"):
        if not graph_pic.get("url") or graph_pic.get("is_silhouette"):
            merged["picture"] = apify_meta["picture"]

    if apify_meta.get("cover") and apify_meta.get("cover").get("source"):
        if not (merged.get("cover") or {}).get("source"):
            merged["cover"] = apify_meta["cover"]
        
    if apify_meta.get("about") and not merged.get("about"):
        merged["about"] = apify_meta["about"]

    merged["resource_type"] = merged.get("resource_type") or apify_meta.get("resource_type") or "page"
    merged["_apify_fallback_used"] = True
    merged["_source"] = (merged.get("_source") or "graph") + "+apify"
    return merged

#ai agent llma 3 
def run_ai_agent_analysis(profile_data):
    """
    Uses Llama 3 (via Groq) to analyze the 'vibe' and content of the page.
    """
    if not groq_client:
        return {"ai_score": 50, "explanation": "AI Agent Disabled (No Key or Library)"}

    try:
        # 1. Prepare the Evidence (From Graph/Apify data)
        bio = (profile_data.get("about") or profile_data.get("description") or "No bio available")[:800]
        name = profile_data.get("name", "Unknown")
        stats = f"Followers: {profile_data.get('followers_count', 0)}, Verified: {profile_data.get('is_verified')}"
        
        # Get post text (if available from Apify scrape)
        recent_posts = " || ".join(profile_data.get("raw_post_texts", [])[:3])
        if not recent_posts:
            recent_posts = "No recent post text available."

        prompt = f"""
        You are a fraud detection expert for Philippine Social Media. 
        Analyze this Facebook Page profile.
        
        DATA:
        Name: {name}
        Bio: {bio}
        Stats: {stats}
        Recent Posts Content: {recent_posts}
        
        TASK:
        Determine if this page shows signs of being a "Poser" (Fake/Scam/Impostor) or Legitimate.
        - Red Flags: Bad grammar, "PM Sent", generic stolen names, claiming to be official without verification.
        - Green Flags: Professional bio, consistent branding, high followers.
        
        OUTPUT JSON ONLY:
        {{
            "ai_score": (0-100),  // 100 = Definitely FAKE/SCAM, 0 = Definitely REAL
            "explanation": "Short 2 sentence reason."
        }}
        """

        # 2. Ask the AI
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a JSON-only fraud analysis API."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
            response_format={"type": "json_object"},
            temperature=0.1 
        )

        # 3. Parse Result
        return json.loads(chat_completion.choices[0].message.content)

    except Exception as e:
        print(f"AI Agent Error: {e}")
        return {"ai_score": 50, "explanation": f"AI Analysis Failed: {str(e)[:50]}..."}

# SCORING 
def compute_poser_score(meta: Dict[str, Any]) -> Dict[str, Any]:
    base = 50; layer1 = 0; layer2 = 0; layer3 = 0
    
    pic = ((meta.get("picture") or {}).get("data") or {})
    has_pic = bool(pic.get("url")) and not bool(pic.get("is_silhouette"))
    has_cover = bool((meta.get("cover") or {}).get("source"))
    followers = int(max(int(meta.get("fan_count") or 0), int(meta.get("followers_count") or 0)))
    posts_count = int(meta.get("recent_posts_count") or 0)
    status = str(meta.get("verification_status") or "").strip().lower()
    verified = bool(meta.get("is_verified")) or status in ("blue_verified", "verified", "meta_verified")
    verified_from_registry = str(meta.get("verification_source") or "").strip().lower() == "verified_registry"
    website = meta.get("website")
    site_has_fb = bool(meta.get("website_has_facebook"))
    site_links_page = bool(meta.get("website_links_to_page"))
    name_val = (meta.get("name") or meta.get("username") or "").strip().lower()
    known_brands = {"one sports", "abs-cbn news", "gma news", "philippine sports commission", "rappler", "inquirer"}

    # L1: Visuals/ Profile Info
    if has_pic: layer1 += 12
    else: layer1 -= 15
    if has_cover: layer1 += 5
    else: layer1 -= 5
    if (meta.get("about") or meta.get("description")): layer1 += 6
    
    # L2: Authority
    if verified:
        layer2 += 40
    else:
        layer2 -= 10
        if site_links_page:
            layer2 += 10
        elif site_has_fb:
            layer2 += 6
        elif website:
            layer2 += 0
    
    # Followers influence (more conservative)
    if followers >= 1000000: layer2 += 18
    elif followers >= 100000: layer2 += 12
    elif followers >= 10000: layer2 += 7
    elif followers >= 1000: layer2 += 3
    elif followers < 100: layer2 -= 5

    if followers == 0 and posts_count > 0 and has_pic:
        layer2 += 8
    if name_val in known_brands:
        layer2 += 15
    
    # Age Logic
    if meta.get("created_time"):
        try:
            dt = datetime.fromisoformat(meta.get("created_time").replace("Z","+00:00"))
            if (datetime.now(timezone.utc) - dt).days > 365: layer2 += 8
        except: pass
    elif followers > 5000: layer2 += 3

    # L3: Activity
    if posts_count > 0: layer3 += 8
    if posts_count >= 5 and followers >= 10000:
        layer3 += 6
    elif posts_count >= 5 and followers < 500:
        layer3 -= 2
    layer3 += meta.get("spam_score", 0)

    # Data availability assessment
    missing_fields = []
    if not has_pic: missing_fields.append("profile_picture")
    if not has_cover: missing_fields.append("cover_photo")
    if not (meta.get("about") or meta.get("description")): missing_fields.append("bio")
    if followers <= 0: missing_fields.append("followers_count")
    if posts_count <= 0: missing_fields.append("recent_posts")
    if not meta.get("link"): missing_fields.append("page_link")
    sparse_env_flags = bool(meta.get("_permissions_restricted")) or bool(meta.get("_apify_failed"))
    if len(missing_fields) >= 4 or sparse_env_flags:
        data_availability = "sparse"
    elif len(missing_fields) >= 2:
        data_availability = "partial"
    else:
        data_availability = "complete"

    # 1. Calculate Rule-Based Score (Math)
    rule_score_raw = max(0, min(100, base + layer1 + layer2 + layer3))
    if has_pic and meta.get("name"):
        rule_score_raw = max(rule_score_raw, 55)
    if verified:
        rule_score_raw = min(rule_score_raw + 25, 100)

    # 2. RUN THE AI AGENT (New Layer)
    def _normalize_ai_explanation(m: Dict[str, Any], r: Dict[str, Any]) -> str:
        exp = str(r.get("explanation") or "").strip()
        if not exp:
            return "AI explanation unavailable."
        followers_local = int(max(int(m.get("fan_count") or 0), int(m.get("followers_count") or 0)))
        pic_local = ((m.get("picture") or {}).get("data") or {})
        has_pic_local = bool(pic_local.get("url")) and not bool(pic_local.get("is_silhouette"))
        has_bio_local = bool(m.get("about") or m.get("description"))
        status_local = str(m.get("verification_status") or "").strip().lower()
        verified_local = bool(m.get("is_verified")) or status_local in ("blue_verified","verified","meta_verified")
        # Apply targeted corrections while preserving the agent's narrative
        try:
            import re as _re
            s = exp
            if verified_local:
                for pat in [r"\bnot verified\b", r"\bunverified\b", r"lacks official verification", r"no verified", r"no verification"]:
                    s = _re.sub(pat, "official verification confirmed", s, flags=_re.I)
                s = _re.sub(r"zero followers|no followers", "audience metrics unavailable", s, flags=_re.I)
                
            if followers_local >= 100000:
                s = _re.sub(r"(low|few|no) follower(s)?", "massive reach", s, flags=_re.I)
            if has_bio_local:
                s = _re.sub(r"(no bio|missing bio|no description)", "bio present with details", s, flags=_re.I)
            if has_pic_local:
                s = _re.sub(r"(no profile picture|default picture|missing profile)", "custom profile image present", s, flags=_re.I)
            if verified_local and ("registry" not in s.lower()):
                s = s.rstrip() + " Registry and badge confirmed."
            return s
        except Exception:
            # Fallback minimal correction
            return (exp + (" Registry and badge confirmed." if verified_local else "")).strip()

    ai_result = run_ai_agent_analysis(meta)
    ai_score = ai_result.get("ai_score", 50)
    ai_reason = _normalize_ai_explanation(meta, ai_result)

    rationale_parts = []
    if verified:
        rationale_parts.append("official verification")
    if followers >= 1000000:
        rationale_parts.append("massive audience")
    elif followers >= 100000:
        rationale_parts.append("large audience")
    elif followers >= 10000:
        rationale_parts.append("active audience")
    elif followers >= 1000:
        rationale_parts.append("active audience")
    if posts_count > 0:
        rationale_parts.append("recent activity")
    if has_pic:
        rationale_parts.append("custom profile image")
    if (meta.get("about") or meta.get("description")):
        rationale_parts.append("detailed bio")
    if site_links_page:
        rationale_parts.append("website links back to page")
    elif site_has_fb:
        rationale_parts.append("website references Facebook")
    if name_val in known_brands:
        rationale_parts.append("recognized brand")
    # remove explicit 'Verdict rationale' string
    ai_trust_score = 100 - ai_score
    
    # Logic: If rules are very certain (>90 or <30), trust the math.
    if rule_score_raw >= 90 or rule_score_raw <= 30:
        final_score = rule_score_raw
    else:
        weight_rules = 0.30
        final_score = (rule_score_raw * weight_rules) + (ai_trust_score * (1 - weight_rules))

    if verified or verified_from_registry:
        final_score = 100

    trust = final_score / 100.0
    
    return {
        "raw_score": int(final_score),
        "trust_score": trust,
        "meets_safety_threshold": trust >= 0.60,
        "layers": {"layer1": layer1, "layer2": layer2, "layer3": layer3, "ai_risk": ai_score, "data_availability": data_availability, "missing_fields": missing_fields},
        "ai_analysis": ai_reason,
        "followers": followers
    }

# Labels
def _score_to_verdict(raw: int) -> str:
    if raw >= 80: return "Low Risk - Likely Authentic / Trusted Source."
    if raw >= 55: return "Moderate Risk - Suspicious / Mixed signals"
    return "High Risk - Likely Poser."

def _get_human_explanation(score: int) -> str:
    if score >= 80: return "Based on the data, this account shows strong signs of authenticity."
    if score >= 55: return "Based on the data, this account presents mixed signals and requires caution."
    return "Based on the data, this looks risky due to missing details or suspicious activity."

def build_response(url: str, meta: Dict[str, Any], score: Dict[str, Any], resolved_id: Optional[str] = None) -> Dict[str, Any]:
    # Extract AI data safely
    ai_risk = score.get("layers", {}).get("ai_risk", 50)
    ai_reason = score.get("ai_analysis") or "AI Analysis Pending"
    ai_rationale = ""
    availability = score.get("layers", {}).get("data_availability", "complete")
    missing = score.get("layers", {}).get("missing_fields", [])
    availability_note = (
        "Data unavailable for public signals; verdict blends AI with limited metadata."
        if availability == "sparse" else (
            "Some signals are missing; verdict blends AI with available metadata."
            if availability == "partial" else ""
        )
    )
    
    return {
        "request": {
            "url": url,
            "hostname": _extract_hostname(url),
            "resolved_id": resolved_id
        },

        #PAGE FACTS
        "metadata": {
            "id": meta.get("id"),
            "name": meta.get("name"),
            "username": meta.get("username"),
            "category": meta.get("category"),
            "fan_count": meta.get("fan_count"),
            "followers_count": meta.get("followers_count"),
            "created_time": meta.get("created_time"),
            "website": meta.get("website"),
            "link": meta.get("link"),
            "about": meta.get("about"),
            "description": meta.get("description"),
            "picture": meta.get("picture"),
            "cover": meta.get("cover"),
            "is_verified": meta.get("is_verified"),
            "verification_status": meta.get("verification_status"),
            "verification_source": meta.get("verification_source"),
            "recent_posts_count": meta.get("recent_posts_count"),
            "post_time_span": meta.get("post_time_span"),
            "resource_type": meta.get("resource_type")
        },

        "analysis": {
            # Final Verdict
            "final_trust_score": int(score.get("trust_score", 0) * 100),
            "verdict": _score_to_verdict(int(score.get("raw_score", 0))),
            "human_explanation": _get_human_explanation(int(score.get("raw_score", 0))),
            "safety_threshold_met": score.get("meets_safety_threshold"),
            "data_availability": availability,
            "availability_note": availability_note,
            
            #"Why - explain "
            "breakdown": {
                "rule_based_score": score.get("raw_score"),
                "ai_agent_trust_score": 100 - ai_risk,
                "ai_explanation": ai_reason,
                "ai_score": ai_risk,
                "ai_verdict": ("Likely Poser" if isinstance(ai_risk, int) and ai_risk >= 70 else ("Likely Authentic" if isinstance(ai_risk, int) and ai_risk <= 30 else "Mixed Signals")),
                "scoring_layers": score.get("layers"),
                "missing_fields": missing,
                "data_availability": availability
            },
            
            "data_source_note": (
                "Verified Registry (confirmed official)"
                if str(meta.get("verification_source") or "").strip().lower() == "verified_registry"
                else (
                    "Apify public scrape (normalized)"
                    if meta.get("_apify_fallback_used")
                    else (
                        "Meta Graph API" + (" (limited access)" if meta.get("_permissions_restricted") else "")
                    )
                )
            )
        }
    }


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "online", 
        "service": "Poser Detection API", 
        "graph_base_url": GRAPH_BASE_URL,
        "token_loaded": bool(META_GRAPH_TOKEN),
        "endpoints": {
            "/api/poser/health": "GET - Health/status",
            "/api/poser/detect": "POST - Analyze Facebook Page/Profile URL"
        }
    })

@app.route("/api/poser/health", methods=["GET"])
def poser_health():
    token_info = _debug_token_info(META_GRAPH_TOKEN) if META_GRAPH_TOKEN else {}
    ap_health = _apify_health()
    is_admin = _require_admin_secret({}) 

    return jsonify({
        "status": "ok",
        "firebase_connected": bool(db),
        "admin_mode": is_admin,
        "graph_base_url": GRAPH_BASE_URL,
        "last_graph_error": LAST_GRAPH_ERROR,
        "graph_cache_size": len(GRAPH_CACHE) if is_admin else None,
        "meta_token_loaded": bool(META_GRAPH_TOKEN),
        "meta_token_valid": token_info.get("is_valid"),
        "meta_token_expires_in_days": token_info.get("expires_in_days"),
        "apify_status": ap_health,
        "ai_agent_online": bool(groq_client),
        "ai_agent_reason": AI_AGENT_REASON,
        "ai_agent_details": {"groq_lib_present": groq_lib_present, "groq_key_present": groq_key_present}
    })

@app.route("/api/poser/admin_mark_verified", methods=["POST"])
def admin_mark_verified():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Missing url"}), 400
    if (POSER_ADMIN_SECRET or "").strip():
        if request.headers.get("X-Admin-Secret") != (POSER_ADMIN_SECRET or "").strip() and not _require_admin_secret(data):
            return jsonify({"error": "Forbidden"}), 403
    if not db:
        return jsonify({"error": "Database unavailable"}), 500
    try:
        resolved = _resolve_fb_share_url(url) if "facebook.com/share/" in url else None
        base_url = _normalize_to_page_url(resolved or url)
        target_urls = list({url, base_url, (resolved or url)})
        payload = {
            "_original_url": url,
            "is_verified": True,
            "verification_status": "blue_verified",
            "verification_source": "verified_registry",
            "last_updated": firestore.SERVER_TIMESTAMP
        }
        updated = []
        for u in target_urls:
            try:
                did = _get_cache_key(u)
                db.collection(RAW_DATA_COLLECTION).document(did).set(payload, merge=True)
                updated.append(did)
            except Exception:
                continue
        try:
            for u in target_urls:
                did = _get_cache_key(u)
                meta_doc = db.collection(RAW_DATA_COLLECTION).document(did).get()
                meta = (meta_doc.to_dict() or {}) if meta_doc.exists else {}
                meta["is_verified"] = True
                meta["verification_status"] = "blue_verified"
                meta["verification_source"] = "verified_registry"
                try:
                    score = compute_poser_score(meta)
                except Exception:
                    score = {"raw_score": 95, "trust_score": 0.95, "layers": {"ai_risk": 0}, "ai_analysis": "Verified account with official signals. Registry and badge confirmed.", "followers": int(meta.get("followers_count") or 0)}
                try:
                    res = build_response(u, meta, score, resolved_id=meta.get("id"))
                except Exception:
                    res = {
                        "metadata": {"is_verified": True, "verification_status": "blue_verified", "verification_source": "verified_registry"},
                        "analysis": {
                            "final_trust_score": int(score.get("trust_score", 0.95) * 100),
                            "verdict": "Low Risk - Likely Authentic / Trusted Source.",
                            "human_explanation": "Verified Registry: Official page confirmed. Strong signals of authenticity.",
                            "data_source_note": "Verified Registry (confirmed official)"
                        }
                    }
                db.collection(VERDICT_COLLECTION).document(did).set({**res, "analysis": res, "last_updated": firestore.SERVER_TIMESTAMP}, merge=True)
        except Exception:
            pass
        return jsonify({"status": "success", "updated_docs": updated, "targets": target_urls})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/poser/admin_migrate_cache", methods=["POST"])
def admin_migrate_cache():
    data = request.get_json(force=True) or {}
    if not _require_admin_secret(data):
        return jsonify({"error": "Unauthorized"}), 401
    if not db:
        return jsonify({"error": "No database"}), 500
    limit = int(data.get("limit") or 100)
    migrated = []
    skipped = []
    try:
        docs = db.collection(RAW_DATA_COLLECTION).limit(limit).get()
        for d in docs:
            try:
                meta = d.to_dict() or {}
                url = (meta.get("_original_url") or meta.get("link") or "").strip()
                if not url:
                    skipped.append(d.id)
                    continue
                score = compute_poser_score(meta)
                res = build_response(url, meta, score, resolved_id=meta.get("id"))
                did = _get_cache_key(url)
                payload = {**res, "last_updated": firestore.SERVER_TIMESTAMP, "analysis": res}
                db.collection(VERDICT_COLLECTION).document(did).set(payload, merge=True)
                migrated.append(did)
            except Exception:
                skipped.append(d.id)
                continue
        return jsonify({"migrated_count": len(migrated), "skipped_count": len(skipped), "migrated": migrated[:50]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

#endpoint
@app.route("/api/poser/analyze_full", methods=["POST"])
def poser_analyze_full():
    data = request.get_json(force=True) or {}
    url = (data.get("url") or data.get("id_or_url") or "").strip().strip('`').strip('"').strip("'")
    user_id = data.get('userID') or data.get('userId') or data.get('uid')
    # 2. Check DB status early
    if db is None:
         return jsonify({"error": "Database connection failed at startup."}), 500
    if not url: return jsonify({"error": "Missing url"}), 400
    parsed = parse_url(url)
    if not parsed.get("is_valid"): return jsonify({"error": "Invalid URL"}), 400
    
    if "facebook.com/share/" in url:
        resolved = _resolve_fb_share_url(url)
        if resolved:
            url = resolved
            parsed = parse_url(url)
    fbid = extract_fbid(url)
    registry = None
    try:
        registry = _check_verified_registry(url)
    except Exception:
        registry = None

    # --- 1. check muna first if may existing data na (Fastest) ---
    if db:
        try:
            doc_id = _get_cache_key(url)
            doc = db.collection(VERDICT_COLLECTION).document(doc_id).get()
            if doc.exists:
                existing_data = doc.to_dict() or {}
                try:
                    reg2 = registry or _check_verified_registry(url)
                except Exception:
                    reg2 = None
                if reg2 and (reg2.get("verified") or reg2.get("is_verified") or reg2.get("is_verified_source")):
                    try:
                        resp = existing_data.get("analysis") or existing_data
                        if isinstance(resp, dict):
                            md = resp.get("metadata") or {}
                            md["is_verified"] = True
                            md["verification_status"] = "blue_verified"
                            md["verification_source"] = "verified_registry"
                            resp["metadata"] = md
                            an = resp.get("analysis") or {}
                            an["data_source_note"] = "Verified Registry (confirmed official)"
                            try:
                                raw_score = int((an.get("breakdown") or {}).get("rule_based_score") or 80)
                            except Exception:
                                raw_score = 80
                            final_score = 100
                            an["final_trust_score"] = final_score
                            an["verdict"] = "Low Risk - Likely Authentic / Trusted Source."
                            an["human_explanation"] = "Verified Registry: Official page confirmed. Strong signals of authenticity."
                            bd = an.get("breakdown") or {}
                            try:
                                new_ai = run_ai_agent_analysis(md)
                                ai_score = int(new_ai.get("ai_score", 50))
                                ai_expl = _normalize_ai_explanation(md, new_ai)
                                bd["ai_score"] = ai_score
                                bd["ai_agent_trust_score"] = 100 - ai_score
                                bd["ai_explanation"] = ai_expl
                                bd["ai_verdict"] = ("Likely Poser" if ai_score >= 70 else ("Likely Authentic" if ai_score <= 30 else "Mixed Signals"))
                            except Exception:
                                bd["ai_explanation"] = "Verified account with official signals. Registry and badge confirmed."
                                bd["ai_agent_trust_score"] = 100
                                bd["ai_verdict"] = "Likely Authentic"
                            bd["rule_based_score"] = 100
                            an["breakdown"] = bd
                            resp["analysis"] = an
                            db.collection(VERDICT_COLLECTION).document(doc_id).set({**resp, "analysis": resp, "last_updated": firestore.SERVER_TIMESTAMP}, merge=True)
                            return jsonify(resp)
                    except Exception:
                        pass
                if existing_data.get("analysis"):
                    return jsonify(existing_data["analysis"])
                if existing_data.get("classification"):
                    return jsonify(existing_data)
        except Exception:
            pass

    #raw data here
    meta = None
    if db:
        try:
            doc = db.collection(RAW_DATA_COLLECTION).document(doc_id).get()
            if doc.exists:
                cached = doc.to_dict() or {}
                is_complete = (
                    bool(cached.get("name")) and 
                    (int(cached.get("followers_count") or 0) > 0 or int(cached.get("fan_count") or 0) > 0) and
                    (cached.get("about") or cached.get("description"))
                )
                
                if is_complete:
                    meta = cached
                    if meta.get("id"): fbid = meta.get("id")
                else:
                    print(f"Ignoring incomplete cache for: {url}")
        except Exception:
            pass

    #scraping new datas here
    if not meta:
        base_page_url = _normalize_to_page_url(url)
        meta = fetch_metadata(extract_fbid(base_page_url))
        
        graph_pic = ((meta.get("picture") or {}).get("data") or {})
        has_bad_pic = (not graph_pic.get("url")) or bool(graph_pic.get("is_silhouette"))
        
        # if graph api failed to get the data, apify will do it
        needs_fallback = (
            bool(meta.get("_permissions_restricted")) or 
            (meta.get("fan_count") in (None, 0) and meta.get("followers_count") in (None, 0)) or
            has_bad_pic or
            (not ((meta.get("cover") or {}).get("source"))) or
            (not (meta.get("name") or meta.get("username"))) or
            (not meta.get("link")) or
            (meta.get("recent_posts_count", 0) == 0) or
            (not meta.get("about") and not meta.get("description"))
        )

        if FORCE_APIFY:
            needs_fallback = True

        if needs_fallback:
            try:
                meta["_apify_attempted"] = True
            except Exception:
                pass
            ap_meta = run_apify_scraper(base_page_url)
            if ap_meta:
                meta = _merge_apify_into_meta(meta, ap_meta)
                if not fbid and meta.get("id"):
                    fbid = meta.get("id")
            else:
                try:
                    meta["_apify_failed"] = True
                except Exception:
                    pass

    try:
        flags = _check_website_reciprocity(meta.get("website"), base_page_url or meta.get("link"), meta.get("username"))
        meta.update(flags)
    except Exception:
        pass

    # tas check kung existing sa verified_registry natin
    try:
        reg = registry or _check_verified_registry(url)
        if reg and (reg.get("verified") or reg.get("is_verified") or reg.get("is_verified_source")):
            meta["is_verified"] = True
            meta["verification_status"] = "blue_verified"
            meta["verification_source"] = "verified_registry"
    except Exception:
        pass
    
    if not meta.get("username") and fbid:
        meta["username"] = fbid
    elif not meta.get("username") and meta.get("link"):
          meta["username"] = extract_fbid(meta["link"])

    score = compute_poser_score(meta)
    res = build_response(url, meta, score, resolved_id=fbid)

    # --- SAVE FINAL VERDICT AND USER ID ---
    if db:
        try:
            doc_id = _get_cache_key(url)
            
            # Save Raw Data
            meta_to_save = meta.copy()
            meta_to_save['_cached_at'] = datetime.now(timezone.utc).isoformat()
            meta_to_save['_original_url'] = url
            db.collection(RAW_DATA_COLLECTION).document(doc_id).set(meta_to_save, merge=True)
            
            # Save Final Verdict (to poser_detections collection)
            payload = {
                **res, 
                "last_updated": firestore.SERVER_TIMESTAMP, 
                "analysis": res,
                "userID": user_id, 
                "feedback": { 
                    "agreeCount": 0,
                    "disagreeCount": 0,
                    "voters": {}
                }
            }
            db.collection(VERDICT_COLLECTION).document(doc_id).set(payload, merge=True)
            
        except Exception as e: 
             print(f"Failed to cache result: {e}")

    return jsonify(res)

def _extract_hostname(u: Optional[str]) -> Optional[str]:
    try:
        if not u: return None
        from urllib.parse import urlparse
        host = (urlparse(str(u)).netloc or "").lower()
        return host
    except Exception:
        return None

def _check_website_reciprocity(website_url: Optional[str], fb_link: Optional[str], username: Optional[str]) -> Dict[str, bool]:
    flags = {"website_has_facebook": False, "website_links_to_page": False}
    try:
        if not website_url: return flags
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"}
        r = requests.get(str(website_url), headers=headers, timeout=3)
        html = (r.text or "").lower()
        if "facebook.com" in html:
            flags["website_has_facebook"] = True
            uname = (username or "").strip().lower()
            if uname and ("facebook.com/" + uname) in html:
                flags["website_links_to_page"] = True
            else:
                link = (fb_link or "").strip().lower()
                if link and link.replace("https://www.", "https://") in html:
                    flags["website_links_to_page"] = True
    except Exception:
        pass
    return flags

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(_load_env_var("PORT", "5001")), debug=True)
