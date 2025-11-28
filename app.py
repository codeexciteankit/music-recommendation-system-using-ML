from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# ---------- LOAD & PREPARE DATA ----------
songs = pd.read_csv("data.csv")

feature_cols = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

# Keep rows that have all needed features
songs_clean = songs.dropna(subset=feature_cols).reset_index(drop=True)

X = songs_clean[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(X_scaled)

# ---------- ROUTES ----------

@app.route("/")
def home():
    # This will serve templates/index.html
    return render_template("index.html")


@app.route("/recommend", methods=["GET"])
def recommend():
    song_query = request.args.get("song", "").strip()

    if not song_query:
        return jsonify({"error": "No song name provided"}), 400

    # Find songs whose name contains the query text
    matches = songs_clean[songs_clean["name"].str.contains(song_query, case=False, na=False)]

    if matches.empty:
        return jsonify({"error": "Song not found in dataset"}), 404

    # Take the first matching song as the reference
    idx = matches.index[0]

    distances, indices = model.kneighbors(
        X_scaled[idx].reshape(1, -1),
        n_neighbors=7  # 1 (itself) + 6 recommendations
    )

    recommendations = []
    for dist, ind in zip(distances[0][1:], indices[0][1:]):  # skip itself
        row = songs_clean.loc[ind]
        recommendations.append({
            "song": str(row["name"]),
            "artist": str(row["artists"])
        })

    return jsonify({
        "input": str(matches.iloc[0]["name"]),
        "recommendations": recommendations
    })


if __name__ == "__main__":
    app.run(debug=True)
