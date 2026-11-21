# Whale Watch Planner (Australia)

Live app: (add your Streamlit Community Cloud URL here)

## What it does
Pick a month, or a base city and travel dates, and see where along the Australian
coast humpback whales have been easiest to spot. For trips starting within 7 days,
the app also shows a wave and wind forecast for the top spots.

## Why I built it
(Write your own reason here, e.g. a trip to Australia where you hoped to see whales.)

## Data
- Sightings: Atlas of Living Australia (ALA), *Megaptera novaeangliae*. DOI: (add the DOI shown at download)
- Sea surface temperature: NOAA OISST v2.1, monthly mean, 0.25 degree
- Forecasts: Open-Meteo Marine API and Forecast API

## Method
1. Clean the records (missing values, out-of-range coordinates, duplicates, unreadable dates)
   and count them on a 0.25 degree grid (about 25 km) by month.
   Records after each step: (add your counts)
2. Build presence/absence data: a grid cell with at least one record in any month is treated
   as "observed"; each observed cell, year and month is 1 if there is a record, else 0.
3. Features: latitude, longitude, month as sine and cosine, their interactions with latitude,
   and sea surface temperature.
4. Train on years before 2020 and test on 2020 onwards. Compare a baseline
   (past frequency per cell and month), logistic regression and LightGBM by AUC and calibration.
5. Predict for every cell and month with climatological sea surface temperature.

| Model | Test AUC |
|---|---|
| Baseline (past frequency) | (fill in) |
| Logistic regression | (fill in) |
| LightGBM | (fill in) |

## Limitations
- Presence-only records. "Absence" is approximated, and records cluster where observers are,
  so the index shows relative ease of spotting, not the probability of an encounter.
- Distances are straight lines, not travel distances.
- The forecast outlook (Good / Fair / Poor) is a rule of thumb, not a validated model.
- Some records are licensed for non-commercial use only.

## Next steps
(e.g. target-group background points, bathymetry, other species such as southern right whales)

## How to run
```
pip install -r requirements-pipeline.txt
python pipeline/step1_prepare_sightings.py data/raw/records.zip
python pipeline/step2_sst.py
python pipeline/step3_build_dataset.py
python pipeline/step4_train_logit.py
python pipeline/step5_train_lgbm.py
python pipeline/step6_predict.py
pip install -r requirements.txt
streamlit run app.py
```
