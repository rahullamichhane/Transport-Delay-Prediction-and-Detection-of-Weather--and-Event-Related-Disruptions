# Algorithm: Public Transport Delay Detection (Toronto TTC)

## Overview
The system detects whether a public transport vehicle (Bus, Streetcar, or Subway) will be
significantly delayed (>= 10 minutes) using live-streaming records enriched with weather
and event data. Six machine learning models are trained and compared.

---

## Phase 1 — Vehicle Class Hierarchy (Object-Oriented Design)

### Base class: `Vehicle`
Attributes: `vehicle_id`, `route_id`, `stop_id`, `route_type`, `temperature`, `humidity`,
`weather_condition`, `event_type`, `hour`, `day_of_week`, `is_peak_hour`, `is_weekend`.

Methods:
- `base_delay()` — abstract; returns type-specific base delay in minutes
- `weather_impact()` — delay added by weather condition
- `event_impact()` — delay added by nearby events
- `peak_impact()` — delay added during rush hours (7–9, 16–18)
- `weekend_impact()` — delay added on weekends
- `winter_impact(month)` — extra delay for snow in Dec/Jan/Feb
- `compute_delay(month, noise)` — total delay = sum of all impacts + noise
- `is_significantly_delayed(delay)` — True if delay >= 10 min
- `to_record(...)` — convert to a streaming record dict

### Subclasses
| Class | Base Delay | Notes |
|-------|-----------|-------|
| `Bus` | 5 min | Surface route, moderate traffic impact |
| `Streetcar` | 7 min | Surface route, highest traffic impact |
| `Subway` | 3 min | Underground, weather impact reduced by 60% |

---

## Phase 2 — Data Generation (Live Streaming)

The `TransportDataGenerator` creates records one at a time to simulate a live feed:

1. Pick a random vehicle type (Bus / Streetcar / Subway)
2. Instantiate the matching vehicle class with random attributes
3. Set `hour`, `day_of_week`, `is_peak_hour`, `is_weekend` from the stream timestamp
4. Compute `delay_minutes = vehicle.compute_delay(month, noise)`
5. Set `delay_label = 1` if `delay_minutes >= 10`, else `0`
6. Yield the record; repeat for `n_records`

**Historical mode:** `generate_historical(n)` produces a static DataFrame for training.
**Live mode:** `stream_records(n)` yields records one at a time (simulated real-time feed).

---

## Phase 3 — Preprocessing

1. Drop identifier columns: `record_id`, `timestamp`, `vehicle_id`, `route_id`, `stop_id`, `delay_minutes`
2. Target: `delay_label`
3. Numeric features → median impute + StandardScaler:
   `hour`, `day_of_week`, `is_peak_hour`, `is_weekend`, `temperature`, `humidity`
4. Categorical features → most-frequent impute + OneHotEncoder:
   `vehicle_class`, `route_type`, `weather_condition`, `event_type`
5. Stratified train/test split (80/20)

---

## Phase 4 — Model Training (6 ML Models)

For each model, build a `Pipeline(preprocessor, classifier)` and fit on training data:

| # | Model | Class |
|---|-------|-------|
| 1 | Logistic Regression | `LogisticRegression` |
| 2 | Random Forest | `RandomForestClassifier` |
| 3 | Gradient Boosting | `GradientBoostingClassifier` |
| 4 | K-Nearest Neighbors | `KNeighborsClassifier` |
| 5 | Naive Bayes | `GaussianNB` |
| 6 | XGBoost | `XGBClassifier` (optional) |

Evaluation metrics computed on the test set:
- Accuracy, Precision, Recall, F1-score, ROC-AUC
- Confusion matrix

All pipelines are saved to `output/models/` as `.pkl` files.

---

## Phase 5 — Live Streaming Detection

1. Load all trained pipelines + preprocessor from `output/models/`
2. Create a `StreamingPredictor` with a rolling window (default 200 records)
3. Register every model with the predictor
4. For each live record:
   a. Build a one-row DataFrame from the record
   b. Run `pipeline.predict(row)` for every model → predicted delay label
   c. Append record + predictions to the predictor history
5. Track per-model predicted delay rate over the rolling window

---

## Phase 6 — Output Generation (saved to `output/`)

| Folder | Contents |
|--------|----------|
| `output/models/` | Trained pipeline `.pkl` files + preprocessor |
| `output/tables/` | `model_comparison.csv` — Accuracy, Precision, Recall, F1, ROC-AUC |
| `output/images/` | `model_comparison.png`, per-model confusion matrices, ROC curves, delay-by-vehicle-class, delay-label distribution |
| `output/detections/` | `delay_detections.csv` — every streamed record with true label + per-model predictions |

---

## Phase 7 — Dashboard (Streamlit)

The dashboard loads pipelines from `output/models/` and:
- Starts a live stream of TTC records
- Shows KPIs: records streamed, delayed vehicles, delay rate, average delay
- Shows per-model predicted delay rates
- Charts: delay minutes over time (colored by vehicle class), predicted delay share by model
- Vehicle class breakdown: counts and delay rates for Bus / Streetcar / Subway
- Recent detections table with vehicle badges and model predictions

---

## Decision Rule

A vehicle is classified as **DELAYED** when `delay_minutes >= 10` (i.e. `delay_label == 1`).
Each ML model independently predicts this label for every live record.
