# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Music Recommendation Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- LOAD DATA WITH CACHING ----------
@st.cache_data
def load_data():
    songs = pd.read_csv("data.csv")
    by_year = pd.read_csv("data_by_year.csv")
    by_genre = pd.read_csv("data_by_genres.csv")
    return songs, by_year, by_genre

songs, by_year, by_genre = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("📊 Music Analytics Dashboard")
st.sidebar.write("Use the controls below to explore the datasets.")

section = st.sidebar.radio(
    "Select view:",
    ["Overview", "Year-wise Trends", "Genre Insights", "Song-level Feature Analysis"]
)

# Common feature list from main dataset
feature_cols = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence"
]

# ---------- MAIN SECTIONS ----------

if section == "Overview":
    st.title("🎵 Music Recommendation System – Analytics Overview")

    st.markdown("""
    This dashboard visualizes the **audio features** and **trends** present in the music dataset  
     used by the recommendation system.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Songs", f"{len(songs):,}")
    col2.metric("Years Covered", f"{int(by_year['year'].min())} - {int(by_year['year'].max())}")
    col3.metric("Genres Count", f"{by_genre['genres'].nunique():,}")

    st.subheader("Sample of Main Dataset")
    st.dataframe(songs.head(20))

    st.subheader("Correlation Between Features")
    corr = songs[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


elif section == "Year-wise Trends":
    st.title("📈 Year-wise Feature Trends")

    st.markdown("Explore how musical features have changed over time.")

    # Filter by year range
    min_year = int(by_year["year"].min())
    max_year = int(by_year["year"].max())
    year_range = st.slider("Select year range:", min_year, max_year, (min_year, max_year), step=1)

    by_year_filtered = by_year[(by_year["year"] >= year_range[0]) & (by_year["year"] <= year_range[1])]

    # Choose features to display
    trend_features = st.multiselect(
        "Select features to plot:",
        ["danceability", "energy", "valence", "acousticness", "tempo", "loudness"],
        default=["danceability", "energy", "valence"]
    )

    if trend_features:
        fig, ax = plt.subplots(figsize=(10, 6))
        for feat in trend_features:
            if feat in by_year_filtered.columns:
                ax.plot(by_year_filtered["year"], by_year_filtered[feat], marker="o", label=feat)
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Value")
        ax.set_title("Year-wise Trends")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Select at least one feature to display trend.")

    st.subheader("Raw Year-wise Data")
    st.dataframe(by_year_filtered)


elif section == "Genre Insights":
    st.title("🎼 Genre-wise Insights")

    st.markdown("Analyze how different genres behave in terms of popularity, energy, danceability, etc.")

    # Top genres by popularity
    top_n = st.sidebar.slider("Number of top genres by popularity:", 5, 30, 10)
    genre_pop = by_genre.groupby("genres")["popularity"].mean().sort_values(ascending=False).head(top_n)

    st.subheader(f"Top {top_n} Genres by Popularity")
    st.bar_chart(genre_pop)

    st.subheader("Danceability vs Energy (Top Genres)")
    selected_genres = genre_pop.index.tolist()
    subset = by_genre[by_genre["genres"].isin(selected_genres)]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        subset["danceability"],
        subset["energy"],
        s=subset["popularity"] * 5,
        alpha=0.7
    )
    ax.set_xlabel("Danceability")
    ax.set_ylabel("Energy")
    ax.set_title("Danceability vs Energy (Bubble size = Popularity)")

    # Add text labels
    for _, row in subset.iterrows():
        ax.text(row["danceability"], row["energy"], row["genres"], fontsize=8)

    st.pyplot(fig)

    st.subheader("Genre Data")
    st.dataframe(subset)


elif section == "Song-level Feature Analysis":
    st.title("🎚 Song-level Feature Analysis")

    st.markdown("Explore distributions and relationships between audio features of individual tracks.")

    # Select a feature to plot distribution
    feat = st.selectbox("Select a feature for distribution:", feature_cols)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Distribution of {feat}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=songs, x=feat, kde=True, bins=30, ax=ax)
        ax.set_xlabel(feat)
        st.pyplot(fig)

    with col2:
        st.subheader(f"{feat} vs Popularity")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(songs[feat], songs["popularity"], alpha=0.3)
        ax.set_xlabel(feat)
        ax.set_ylabel("Popularity")
        st.pyplot(fig)

    st.subheader("Energy vs Danceability Scatter")
    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        songs["danceability"], songs["energy"],
        c=songs["valence"], cmap="viridis", alpha=0.5
    )
    ax.set_xlabel("Danceability")
    ax.set_ylabel("Energy")
    cb = fig.colorbar(sc)
    cb.set_label("Valence (Happiness)")
    st.pyplot(fig)

    st.subheader("Raw Songs Data (Sample)")
    st.dataframe(songs[["name", "artists", "year", "popularity"] + feature_cols].head(50))
