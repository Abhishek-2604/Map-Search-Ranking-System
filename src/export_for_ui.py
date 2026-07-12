"""
Exports:
  1. models/model_min.json  - stripped LightGBM trees for JS inference (same as before)
  2. static/restaurants.json - the real restaurant catalog, ready for the UI:
     real name, real coordinates (converted to km offsets from a fixed city-center
     anchor), real cuisines, real price level, real rating, real num_ratings,
     and a compact weekly open-hours schedule for live is-open computation.
"""
import json
import numpy as np

DATA = "/home/claude/map_search_ltr/data"
MODELS = "/home/claude/map_search_ltr/models"
STATIC = "/home/claude/map_search_ltr/static"

FEATURES = json.load(open(f"{MODELS}/features.json"))


def strip_tree(node):
    if "leaf_value" in node:
        return {"leaf": node["leaf_value"]}
    return {
        "f": node["split_feature"], "t": node["threshold"],
        "l": strip_tree(node["left_child"]), "r": strip_tree(node["right_child"]),
    }


def export_model():
    dumped = json.load(open(f"{MODELS}/lambdamart.json"))
    trees = [strip_tree(t["tree_structure"]) for t in dumped["tree_info"]]
    json.dump({"features": FEATURES, "trees": trees}, open(f"{MODELS}/model_min.json", "w"))
    print(f"Exported {len(trees)} trees")


def export_restaurants():
    restaurants = json.load(open(f"{DATA}/restaurants_real.json"))
    lat_c = float(np.mean([r["lat"] for r in restaurants]))
    lng_c = float(np.mean([r["lng"] for r in restaurants]))

    out = []
    for r in restaurants:
        dlat = (r["lat"] - lat_c) * 111.0
        dlng = (r["lng"] - lng_c) * 111.0 * np.cos(np.radians(lat_c))
        out.append({
            "name": r["name"],
            "cuisines": r["cuisines"],
            "x": round(dlng, 3), "y": round(dlat, 3),
            "price_level": r["price_level"],
            "rating": r["rating"],
            "num_ratings": r["num_ratings"],
            "franchise": r["franchise"],
            "schedule": r["schedule"],
        })
    payload = {
        "city": "San Luis Potosi, Mexico (real coordinates, anchored at dataset centroid)",
        "anchor_lat": lat_c, "anchor_lng": lng_c,
        "restaurants": out,
    }
    json.dump(payload, open(f"{STATIC}/restaurants.json", "w"))
    print(f"Exported {len(out)} real restaurants")


if __name__ == "__main__":
    export_model()
    export_restaurants()
