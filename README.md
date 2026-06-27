# VECTOR.FC — Football Player Similarity Engine

A professional-grade scouting tool that answers the question: *"Find me players 
with a similar profile to X."*

Built on StatsBomb open data, powered by K-Means clustering and cosine similarity 
across 4,475 player profiles from 80 competition-seasons. SHAP-style feature 
contribution breakdown explains *why* two players are comparable — not just that 
they are.

**[→ Live demo](https://kwakeel-football-scouting-tool.streamlit.app)**

## What it does
- **Player Profile** — search any of 4,475 players, see their archetype cluster, 
  percentile rankings across 21 features, and their 10 most similar players
- **Player Comparison** — compare any two players, see their cosine similarity 
  score and a feature-by-feature breakdown of what makes them similar and different

## Key result
Busquets → Song at 0.96 cosine similarity. Jordi Alba → Van der Wiel, Evra, 
Digne, Nuno Mendes — profile-based matching working across positions, eras, 
and leagues with no position labels used.

## Stack
Python · StatsBomb open data · K-Means · Cosine similarity · Streamlit
