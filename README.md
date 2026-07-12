# Wayfinder — Map Search Ranking with Learning to Rank (real data, live search)

A small end-to-end project: train a **LambdaMART** learning-to-rank model over
a **real restaurant dataset**, then let people type **real search queries**
into a live, animated browser UI that runs the actual trained model
client-side.

```
map_search_ltr/
├── data/
│   ├── real_geoplaces2.csv        raw UCI dataset (places, real GPS coords)
│   ├── real_chefmozcuisine.csv    raw UCI dataset (real cuisine tags)
│   ├── real_chefmozhours4.csv     raw UCI dataset (real weekly opening hours)
│   ├── real_rating_final.csv      raw UCI dataset (real consumer ratings)
│   ├── prepare_real_data.py       cleans/merges the above -> restaurants_real.json
│   ├── generate_sessions.py       simulates search sessions over the real catalog
│   └── restaurants_real.json      70 real restaurants, cleaned
├── src/
│   ├── train.py                    trains the LambdaMART ranker (LightGBM)
│   └── export_for_ui.py            exports trained trees + real catalog for the UI
├── models/                         trained model (native + JSON + minimal JSON)
├── static/restaurants.json         real restaurant catalog, UI-ready
├── ui_template.html                UI source (placeholders for injected data)
└── index.html                       the finished, self-contained demo — open this
```

## What's real vs. simulated

| Piece | Source |
|---|---|
| Restaurant names, GPS coordinates, cuisines, prices, weekly hours | **Real.** UCI "Restaurant & Consumer Data" (Vargas-Govea et al., 2011), San Luis Potosí, Mexico. |
| Star ratings, review counts | **Real.** Aggregated from real diners' real ratings in the same dataset. |
| Search queries and relevance labels used to *train* the ranker | **Simulated.** No public query-log dataset exists for this restaurant catalog, so training sessions (user location, query, relevance judgment) are generated the same way a company would bootstrap a ranker before it has click logs — see `generate_sessions.py`. |
| The query *you* type into the demo | **Real, yours.** Matched live against the real names/cuisines. |

## Run it yourself

```bash
pip install lightgbm scikit-learn pandas numpy
python3 data/prepare_real_data.py   # clean/merge real UCI CSVs -> restaurants_real.json
python3 data/generate_sessions.py   # simulate training sessions over the real catalog
python3 src/train.py                # train models/lambdamart.{txt,json}
python3 src/export_for_ui.py        # export models/model_min.json + static/restaurants.json
```

Then rebuild `index.html`:
```bash
python3 -c "
model = open('models/model_min.json', encoding='utf-8').read()
restaurants = open('static/restaurants.json', encoding='utf-8').read()
tpl = open('ui_template.html', encoding='utf-8').read()
open('index.html','w',encoding='utf-8').write(tpl.replace('__MODEL_JSON__', model).replace('__RESTAURANT_JSON__', restaurants))
"
```
Or just open the pre-built `index.html` directly — no server needed.

## The algorithm: LambdaMART

`train.py` trains **LambdaMART** — gradient-boosted trees (LightGBM's
`LGBMRanker`, `objective="lambdarank"`) with a pairwise/listwise ranking loss:
it learns from pairs of candidates *within the same query* ("A should outrank
B") with gradients weighted by how much swapping them would change NDCG.
That directly optimizes ranking order, unlike pointwise regression on
relevance. Rows are grouped by `query_id`, split by query (not row) to avoid
leakage, and evaluated with NDCG@3/5/10 on held-out queries.

Result: **NDCG@5 ≈ 0.955**. `text_match` and `rating` dominate feature
importance, followed by `is_open` and `distance_km` — sensible for a
restaurant search ranker.

## The UI (`index.html`)

Trained trees are stripped to `{split_feature, threshold, children,
leaf_value}` and a small JS function walks them and sums leaf values — real
model inference in the browser, not a mock.

- **Type any real search** ("tacos", "pizza", "bar", "breakfast", a
  restaurant's own name…) — the query is matched live against each
  restaurant's real name and cuisine tags to produce a `text_match` feature,
  exactly like the `text_match` feature the model was trained on.
- **Pick a time of day** (Morning / Lunch / Dinner / Late night) — "open now"
  is computed from each restaurant's real weekly hours for *today's* actual
  weekday, so closed places get pushed down or greyed out live.
- Results **re-rank with a FLIP animation** as you type (debounced ~120ms) —
  cards and map pins smoothly slide to their new position rather than
  jumping, so you can see the ranking change happen rather than just the
  end state.
- A feature-importance panel shows what the model actually learned to weight
  during training.

This version is intentionally minimal: no filter sliders, no ranking-mode
toggle — the model handles the tradeoffs, and the only inputs are a real
search box and a time-of-day picker.

## Ideas to extend

- Add more real signal: parking/accessibility/ambience fields already exist
  in the raw UCI data but aren't used yet.
- Try `objective="rank_xendcg"` or a neural ranker and compare NDCG.
- If you ever get real click logs for a catalog like this, swap
  `generate_sessions.py`'s simulated labels for real position-bias-corrected
  click data.
- Add a second real city from the same dataset (Cuernavaca has 16 restaurants)
  and let people switch between them.
