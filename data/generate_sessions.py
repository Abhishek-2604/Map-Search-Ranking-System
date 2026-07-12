"""
Generates LTR training sessions using the REAL restaurants (data/restaurants_real.json).
The restaurants, their coordinates, cuisines, prices, hours and ratings are all real.
What's simulated here is the *search session* layer: a user location, a typed
query, and a relevance label — because no public query-log dataset exists for
this restaurant set. This mirrors how a real company would bootstrap a ranker
before they have click logs: known catalog + simulated/heuristic sessions.
"""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
DATA = "/home/claude/map_search_ltr/data"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def km_offset(lat1, lng1, lat2, lng2):
    dlat = (lat2 - lat1) * 111.0
    dlng = (lng2 - lng1) * 111.0 * np.cos(np.radians(lat1))
    return float(np.sqrt(dlat ** 2 + dlng ** 2))


def is_open_at(schedule, day, hour):
    if day not in schedule:
        return 0
    oh, ch = schedule[day]
    if oh <= ch:
        return int(oh <= hour < ch)
    return int(hour >= oh or hour < ch)


def main():
    restaurants = json.load(open(f"{DATA}/restaurants_real.json"))
    all_cuisines = sorted({c for r in restaurants for c in r["cuisines"]})
    lats = [r["lat"] for r in restaurants]
    lngs = [r["lng"] for r in restaurants]
    lat_c, lng_c = float(np.mean(lats)), float(np.mean(lngs))

    n_sessions = 500
    rows = []
    for qid in range(n_sessions):
        # simulate a user location near the city center (real coordinate bounding box)
        user_lat = lat_c + rng.normal(0, 0.01)
        user_lng = lng_c + rng.normal(0, 0.01)
        query_cuisine = all_cuisines[rng.integers(0, len(all_cuisines))]
        day = DAYS[rng.integers(0, 7)]
        hour = int(rng.integers(7, 23))
        price_pref = int(rng.integers(1, 4))  # session's implicit price sensitivity

        for r in restaurants:
            dist = km_offset(user_lat, user_lng, r["lat"], r["lng"])
            if dist > 6 and rng.uniform() > 0.15:
                continue  # retrieval stage would drop most far-away places

            cuisine_hit = query_cuisine in r["cuisines"]
            text_match = 1.0 if cuisine_hit else float(rng.uniform(0, 0.2))
            open_flag = is_open_at(r["schedule"], day, hour)
            price_match = 1.0 - abs(r["price_level"] - price_pref) / 2.0
            popularity = float(np.log1p(r["num_ratings"]))

            utility = (
                3.0 * text_match
                + 1.1 * (r["rating"] - 3.0)
                + 0.5 * np.tanh(popularity / 2.0)
                + 0.4 * price_match
                - 0.85 * np.log1p(dist)
                - (2.3 if not open_flag else 0.0)
                + rng.normal(0, 0.35)
            )
            rows.append({
                "query_id": qid, "query_cuisine": query_cuisine, "day": day, "hour": hour,
                "place_id": r["place_id"], "name": r["name"],
                "text_match": round(text_match, 3), "distance_km": round(dist, 3),
                "rating": r["rating"], "num_ratings": r["num_ratings"],
                "popularity": round(popularity, 3), "is_open": open_flag,
                "price_level": r["price_level"], "price_match": round(price_match, 3),
                "_utility": utility,
            })

    df = pd.DataFrame(rows)
    df = df[df.groupby("query_id")["place_id"].transform("count") >= 2]
    ranks = df.groupby("query_id")["_utility"].rank(pct=True)
    df["relevance"] = pd.cut(ranks, bins=[0, 0.55, 0.8, 0.95, 1.0],
                              labels=[0, 1, 2, 3], include_lowest=True).astype(int)
    df = df.drop(columns=["_utility"])
    df.to_csv(f"{DATA}/real_search_sessions.csv", index=False)
    print(f"Generated {len(df)} rows across {df.query_id.nunique()} sessions on {len(restaurants)} real restaurants")
    print(df["relevance"].value_counts().sort_index())


if __name__ == "__main__":
    main()
