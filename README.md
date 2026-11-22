# Whale Watching Planner (Australia)

**Live app:** https://whale-watching-australia.streamlit.app

Pick a month, or a base city and travel dates, and see where along the Australian coast
humpback whales have been easiest to spot. For trips starting within 7 days, the app also
shows a wave and wind forecast for the top spots. The interface is available in English and Japanese.

## Why I built it
I love Australia, and on a trip there I hoped to see whales but missed them.
I wanted a tool that tells travellers where and when they are most likely to succeed,
and I used it as an end-to-end project in data cleaning, statistics and machine learning.

## Data
- **Sightings:** Atlas of Living Australia (ALA), *Megaptera novaeangliae*, downloaded 22 Nov 2026
  with the "ALA General" data quality profile (52,677 of 71,800 records).
  DOI: https://doi.org/10.26197/ala.c9cc0ddb-e9c3-4b0b-abec-2e1a94909728
- **Sea surface temperature:** NOAA OISST v2.1, monthly mean, 0.25 degree, Jan 2000 to Oct 2026
- **Forecasts:** Open-Meteo Marine API and Forecast API
- **Coastline:** Natural Earth 1:10m land polygons (public domain)

Some sighting records are licensed for non-commercial use only. This project is a
non-commercial portfolio project and credits all sources.

## Data cleaning
| Step | Records |
|---|---|
| Loaded | 52,677 |
| Missing coordinates or date removed | 52,051 |
| Outside the Australian study area removed | 50,029 |
| Duplicates (same place and same date) removed | 32,007 |
| Placeholder dates (1 January) removed | 31,980 |

Two checks shaped the cleaning:
- **Placeholder dates.** 27 of 64 January records fell on 1 January (about 2 would be expected),
  so 1 January records were treated as unknown dates and removed.
- **Silent download failure.** The first sea temperature download returned all zeros without an error.
  The pipeline now downloads one year at a time and rejects zeros and impossible values.

## Method
1. Count records on a 0.25 degree grid (about 25 km) by month.
2. Build presence/absence data: a grid cell with at least one record is treated as "observed";
   each observed cell, year (2000 onwards) and month is 1 if there is a record, else 0
   (303,968 rows, 1.4% positive).
3. Features: latitude, longitude, month as sine and cosine, their interactions with latitude,
   and sea surface temperature.
4. Train on years before 2020 and test on 2020 onwards.
5. Predict for every cell and month with climatological (average) sea surface temperature.
6. Place each map marker at the average position of the records in its cell. If that falls on land
   (about 10% of cells), move it to the nearest point about 1.5 km offshore using Natural Earth land polygons.

## Results
| Model | Test AUC |
|---|---|
| Baseline (past frequency per cell and month) | 0.780 |
| Logistic regression | 0.765 |
| LightGBM | **0.886** |

- Because only 1.4% of rows are positive, accuracy would be misleading, so models are compared by AUC.
- Logistic regression lost to the baseline: a single linear equation cannot capture
  differences between individual bays and survey sites. Its coefficients still show
  the expected seasonal peak in July and August.
- LightGBM beat the baseline by about 0.11 because it can combine location, season and
  temperature, and borrow information from nearby cells.
- The index rewards consistency rather than volume. A spot near Augusta (Western Australia)
  had the most July records (298) but they were concentrated in a few years, so it ranks
  below Sydney, which has records in most years.

## Limitations
- Presence-only records. "Absence" is approximated, and records cluster where observers are,
  so the index shows the relative ease of spotting whales, not the probability of an encounter.
- The test used actual sea temperatures for each year, while the app uses average temperatures,
  so real-world performance may be lower than the test AUC.
- Records for the latest year may be incomplete.
- Distances are straight lines, not travel distances.
- The forecast outlook (Good / Fair / Poor) is a rule of thumb, not a validated model.

## Next steps
- Re-evaluate the models using average sea temperatures, as the app does.
- Check calibration: do spots with a 10% prediction see records about 10% of the time?
- Reduce observer bias with target-group background points (records of other marine species).
- Add bathymetry and other species, such as southern right whales.

## How to run
```
pip install -r requirements-pipeline.txt
python pipeline/step1_prepare_sightings.py data/raw/records.zip
python pipeline/step2_sst.py
python pipeline/step3_build_dataset.py
python pipeline/step4_train_logit.py
python pipeline/step5_train_lgbm.py
python pipeline/step6_predict.py
python pipeline/step7_snap_to_sea.py
pip install -r requirements.txt
streamlit run app.py
```
