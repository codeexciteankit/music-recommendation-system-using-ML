# app.py
import os
import time
import csv
import base64
import requests
from functools import lru_cache
from typing import Optional
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
load_dotenv()   # reads .env into environment variables


# -------------------------
# Configuration / env vars
# -------------------------
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "").strip()

PLAYS_CSV = "plays.csv"   # logs play events / likes

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder="templates")

# -------------------------
# Load & prepare dataset
# -------------------------
songs = pd.read_csv("data.csv")

feature_cols = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

# keep rows that have needed features
songs_clean = songs.dropna(subset=feature_cols).reset_index(drop=True)

# Feature matrix
X = songs_clean[feature_cols].values.astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KNN model (cosine)
model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(X_scaled)

# -------------------------
# Helper: plays logging
# -------------------------
def ensure_plays_csv_exists():
    if not os.path.exists(PLAYS_CSV):
        with open(PLAYS_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "song", "artist", "preview_url", "source", "action", "client_ip"])

def log_play_event(song, artist, preview_url, source="web", action="play"):
    ensure_plays_csv_exists()
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    client_ip = request.remote_addr if request else ""
    with open(PLAYS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ts, song, artist, preview_url or "", source, action, client_ip])

# -------------------------
# Spotify: client credentials caching & search fallback
# -------------------------
_spotify_cache = {"token": None, "expires_at": 0}

def spotify_has_creds():
    return bool(SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET)

def get_spotify_token():
    """Return cached Spotify token (client credentials)."""
    now = time.time()
    if _spotify_cache["token"] and _spotify_cache["expires_at"] - 30 > now:
        return _spotify_cache["token"]
    if not spotify_has_creds():
        return None
    auth = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}"}
    data = {"grant_type": "client_credentials"}
    try:
        r = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data, timeout=8)
        r.raise_for_status()
    except Exception as e:
        app.logger.warning("Spotify token error: %s", e)
        return None
    token = r.json().get("access_token")
    expires_in = int(r.json().get("expires_in", 3600))
    _spotify_cache["token"] = token
    _spotify_cache["expires_at"] = now + expires_in
    return token
@lru_cache(maxsize=4096)
def spotify_search_preview(name: str, artist: Optional[str] = None):
# def spotify_search_preview(name: str, artist: str = None):
    """
    Return dict with preview_url, external_url, image_url for the first Spotify result.
    Cached by function (in-memory).
    """
    token = get_spotify_token()
    if not token:
        return {"preview_url": None, "external_url": None, "image_url": None}
    q = name
    if artist:
        q += f" artist:{artist}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": q, "type": "track", "limit": 1}
    try:
        r = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params, timeout=8)
        r.raise_for_status()
    except Exception as e:
        app.logger.debug("Spotify search failed: %s", e)
        return {"preview_url": None, "external_url": None, "image_url": None}
    items = r.json().get("tracks", {}).get("items", [])
    if not items:
        return {"preview_url": None, "external_url": None, "image_url": None}
    t = items[0]
    preview = t.get("preview_url")  # often 30s mp3 or None
    external = t.get("external_urls", {}).get("spotify")
    images = t.get("album", {}).get("images", [])
    image_url = images[0]["url"] if images else None
    return {"preview_url": preview, "external_url": external, "image_url": image_url}

# -------------------------
# YouTube search endpoint (server-side, needs YOUTUBE_API_KEY)
# -------------------------
# backend: improved youtube_search (paste into app.py, replace old route)
@app.route("/youtube_search")
def youtube_search():
    """
    Server-side YouTube lookup that prefers embeddable videos.
    Returns JSON:
      { videoId: str or None, url: str or None, embeddable: bool, title: str or None }
    """
    query = request.args.get("query", "").strip()
    if not query:
        return jsonify({"error": "no query provided"}), 400
    if not YOUTUBE_API_KEY:
        return jsonify({"error": "YOUTUBE_API_KEY not configured"}), 500

    search_params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": 5,   # try multiple results
        "key": YOUTUBE_API_KEY
    }

    try:
        r = requests.get("https://www.googleapis.com/youtube/v3/search", params=search_params, timeout=8)
        r.raise_for_status()
    except Exception as e:
        app.logger.warning("YouTube search fail: %s", e)
        return jsonify({"error": "youtube search failed"}), 500

    items = r.json().get("items", [])
    if not items:
        return jsonify({"videoId": None}), 200

    # Collect candidate videoIds
    candidates = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        title = it.get("snippet", {}).get("title")
        if vid:
            candidates.append({"videoId": vid, "title": title})

    if not candidates:
        return jsonify({"videoId": None}), 200

    # Call Videos API to check embeddable status for candidates (batch request)
    vid_list = ",".join([c["videoId"] for c in candidates])
    vids_params = {
        "part": "status,contentDetails,snippet",
        "id": vid_list,
        "key": YOUTUBE_API_KEY
    }
    try:
        r2 = requests.get("https://www.googleapis.com/youtube/v3/videos", params=vids_params, timeout=8)
        r2.raise_for_status()
    except Exception as e:
        app.logger.warning("YouTube videos lookup failed: %s", e)
        # fallback: return first candidate (may be unembeddable)
        best = candidates[0]
        return jsonify({"videoId": best["videoId"], "url": f"https://www.youtube.com/watch?v={best['videoId']}", "embeddable": False, "title": best.get("title")}), 200

    vids_items = r2.json().get("items", [])

    # Build map of videoId -> embeddable flag
    emb_map = {}
    for it in vids_items:
        vid = it.get("id")
        status = it.get("status", {})
        emb = status.get("embeddable", False)
        emb_map[vid] = bool(emb)

    # Choose first embeddable candidate
    chosen = None
    for c in candidates:
        if emb_map.get(c["videoId"]):
            chosen = {"videoId": c["videoId"], "title": c.get("title"), "embeddable": True}
            break

    # If none embeddable, pick first candidate but mark embeddable False
    if not chosen:
        first = candidates[0]
        chosen = {"videoId": first["videoId"], "title": first.get("title"), "embeddable": False}

    chosen["url"] = f"https://www.youtube.com/watch?v={chosen['videoId']}"
    return jsonify(chosen), 200


# -------------------------
# Routes
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Query param: song (string)
    Returns:
      {
        "input": "<matched song name>",
        "recommendations": [
          {
            "song": ...,
            "artist": ...,
            "year": ...,
            "popularity": ...,
            "features": {...},
            "similarity": 0..1,
            "preview_url": ...,
            "external_url": ...,
            "image_url": ...
          },
          ...
        ]
      }
    """
    song_query = request.args.get("song", "").strip()
    if not song_query:
        return jsonify({"error": "No song name provided"}), 400

    # find by name contains (case-insensitive)
    matches = songs_clean[songs_clean["name"].str.contains(song_query, case=False, na=False)]
    if matches.empty:
        return jsonify({"error": "Song not found in dataset"}), 404

    # choose first matched row as reference
    idx = matches.index[0]

    # neighbors: n_neighbors = 16 (as in your version)
    distances, indices = model.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=16
    )

    recommendations = []
    # distances[0][0] corresponds to itself (distance 0) — skip it
    for dist, ind in zip(distances[0][1:], indices[0][1:]):
        row = songs_clean.loc[ind]
        # pack features safely
        feats = {}
        for f in feature_cols:
            try:
                val = row.get(f)
                feats[f] = float(val) if (val is not None and not pd.isna(val)) else None
            except Exception:
                feats[f] = None

        # attempt to read preview/external/image from dataset columns if present
        preview = None
        external = None
        image_url = None
        if "preview_url" in songs_clean.columns:
            pv = row.get("preview_url")
            if pv is not None and not pd.isna(pv) and str(pv).strip():
                preview = str(pv).strip()
        if "external_url" in songs_clean.columns:
            ex = row.get("external_url")
            if ex is not None and not pd.isna(ex) and str(ex).strip():
                external = str(ex).strip()
        if "image_url" in songs_clean.columns:
            im = row.get("image_url")
            if im is not None and not pd.isna(im) and str(im).strip():
                image_url = str(im).strip()

        # if missing preview or external, try Spotify fallback (if credentials)
        if spotify_has_creds() and (not preview or not external or not image_url):
            try:
                sp = spotify_search_preview(str(row.get("name", "")), str(row.get("artists", "")))
                if not preview and sp.get("preview_url"):
                    preview = sp.get("preview_url")
                if not external and sp.get("external_url"):
                    external = sp.get("external_url")
                if not image_url and sp.get("image_url"):
                    image_url = sp.get("image_url")
            except Exception as e:
                app.logger.debug("Spotify fallback error for %s: %s", row.get("name"), e)

        rec = {
            "song": str(row.get("name")) if row.get("name") is not None else "",
            "artist": str(row.get("artists")) if row.get("artists") is not None else "",
            "year": int(row.get("year")) if ("year" in row and not pd.isna(row.get("year"))) else None,
            "popularity": int(row.get("popularity")) if ("popularity" in row and not pd.isna(row.get("popularity"))) else None,
            "features": feats,
            "similarity": float(1 - dist),
            "preview_url": preview if preview else None,
            "external_url": external if external else None,
            "image_url": image_url if image_url else None
        }
        recommendations.append(rec)

    # also return the matched input name for display
    input_name = str(matches.iloc[0]["name"])
    return jsonify({"input": input_name, "recommendations": recommendations})

# -------------------------
# Logging endpoint (plays, likes)
# -------------------------
@app.route("/log_play", methods=["POST"])
def log_play():
    """
    Expected JSON body:
      { "song": "...", "artist": "...", "preview_url": "...", "source": "web", "action": "play" }
    Appends a row to plays.csv. Returns {"status":"ok"}.
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    song = payload.get("song")
    artist = payload.get("artist")
    preview = payload.get("preview_url")
    source = payload.get("source", "web")
    action = payload.get("action", "play")
    if not song:
        return jsonify({"error": "Missing song field"}), 400

    try:
        log_play_event(song, artist or "", preview or "", source=source, action=action)
    except Exception as e:
        app.logger.warning("Failed to log play event: %s", e)
    return jsonify({"status": "ok"})

# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    if not spotify_has_creds():
        app.logger.info("Spotify credentials not set. Spotify fallback disabled. Set SPOTIFY_CLIENT_ID & SPOTIFY_CLIENT_SECRET to enable.")
    if not YOUTUBE_API_KEY:
        app.logger.info("YOUTUBE_API_KEY not set. /youtube_search endpoint will return error.")
    app.run(debug=True, host="127.0.0.1", port=5000)
