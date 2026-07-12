"""
Cleans and merges the REAL UCI "Restaurant & Consumer Data" dataset
(Vargas-Govea et al. 2011, via github.com/liyenhsu/restaurant-data-with-consumer-ratings,
originally archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

This is real data: real restaurant names, real GPS coordinates, real cuisine
tags, and real aggregated consumer ratings (from real diners rating real
visits) — collected in San Luis Potosi, Mexico. Only the *search sessions*
used to train the ranker (data/generate_sessions.py) are simulated, since no
public query-log dataset exists for this restaurant set.

Output: data/restaurants_real.json — one record per restaurant.
"""
import pandas as pd
import numpy as np
import json
import re

DATA = "/home/claude/map_search_ltr/data"

CITY = "san luis potosi"


def parse_hours(hours_df, place_id):
    """
    chefmozhours4.csv rows look like:
      placeID, "08:00-19:00;", "Mon;Tue;Wed;Thu;Fri;"
    Return {day: [(open_hr, close_hr), ...]} using 24h integer hours,
    collapsing to first interval per day (good enough for a demo).
    """
    rows = hours_df[hours_df.placeID == place_id]
    schedule = {}
    for _, r in rows.iterrows():
        try:
            span = str(r["hours"]).split(";")[0]
            o, c = span.split("-")
            oh = int(o.split(":")[0])
            ch = int(c.split(":")[0])
        except Exception:
            continue
        for day in str(r["days"]).split(";"):
            day = day.strip()
            if day:
                schedule[day] = [oh, ch]
    return schedule


def is_open_at(schedule, weekday_abbr, hour):
    if weekday_abbr not in schedule:
        return 0
    oh, ch = schedule[weekday_abbr]
    if oh <= ch:
        return int(oh <= hour < ch)
    return int(hour >= oh or hour < ch)  # overnight spans


def main():
    places = pd.read_csv(f"{DATA}/real_geoplaces2.csv")
    cuisine = pd.read_csv(f"{DATA}/real_chefmozcuisine.csv")
    ratings = pd.read_csv(f"{DATA}/real_rating_final.csv")
    hours = pd.read_csv(f"{DATA}/real_chefmozhours4.csv")

    places["city_norm"] = places["city"].astype(str).str.lower().str.strip()
    places = places[places["city_norm"] == CITY].copy()

    # cuisines: multiple rows per place -> list
    cuisine_map = cuisine.groupby("placeID")["Rcuisine"].apply(list).to_dict()

    # real consumer ratings: aggregate mean + count per place (rating is 0/1/2 in source)
    rating_agg = ratings.groupby("placeID")["rating"].agg(["mean", "count"]).reset_index()
    rating_map = {r.placeID: (r.mean, int(r.count)) for r in rating_agg.itertuples()}

    price_map = {"low": 1, "medium": 2, "high": 3}

    out = []
    for _, p in places.iterrows():
        pid = int(p["placeID"])
        cuisines = cuisine_map.get(pid, [])
        mean_r, n_r = rating_map.get(pid, (None, 0))
        # scale real 0-2 rating to a familiar 1-5 star scale; unrated places get a neutral 3.0
        rating_5 = round(1.0 + (mean_r * 2.0), 2) if mean_r is not None else 3.0
        schedule = parse_hours(hours, pid)
        name = str(p["name"]).strip()
        if not name or name == "?":
            continue
        # the source CSV has one row with a corrupted byte (U+FFFD) baked in;
        # a duplicate entry elsewhere in the same file confirms the correct spelling
        name = name.replace("\ufffd", "o")
        out.append({
            "place_id": pid,
            "name": name,
            "lat": float(p["latitude"]),
            "lng": float(p["longitude"]),
            "cuisines": cuisines if cuisines else ["Unspecified"],
            "price_level": price_map.get(str(p["price"]).strip().lower(), 2),
            "alcohol": str(p["alcohol"]),
            "ambience": str(p["Rambience"]),
            "franchise": str(p["franchise"]).strip().lower() == "t",
            "rating": rating_5,
            "num_ratings": n_r,
            "schedule": schedule,  # {day_abbr: [open_hr, close_hr]}
        })

    with open(f"{DATA}/restaurants_real.json", "w") as f:
        json.dump(out, f, indent=1)

    print(f"Kept {len(out)} real restaurants in {CITY}")
    n_rated = sum(1 for r in out if r["num_ratings"] > 0)
    print(f"{n_rated} have at least 1 real consumer rating")
    all_cuisines = sorted({c for r in out for c in r["cuisines"]})
    print(f"{len(all_cuisines)} distinct real cuisine tags:", all_cuisines[:20], "...")


if __name__ == "__main__":
    main()
