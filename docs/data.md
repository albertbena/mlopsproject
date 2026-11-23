# Data

## Source

**NYC Taxi & Limousine Commission (TLC), Trip Record Data**, yellow taxi trips:
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page>. One Parquet file per month, with pickup time,
pickup zone (`PULocationID`, 263 taxi zones), distance and fare.

The raw files are **never committed**. `src/taxi/data/download.py` fetches the months listed in `params.yaml`, and
`dvc repro` rebuilds everything from them. `dvc.lock` records the MD5 of every file, so anyone can check that they
get the same data we did.

## From trips to one row per zone and hour

`src/taxi/data/aggregate.py`:

1. Reads only the two columns we use: `tpep_pickup_datetime` and `PULocationID`.
2. Checks the schema: a missing column stops the pipeline. An unknown column, such as `cbd_congestion_fee` added
   in 2025, only prints a warning, because we do not use it.
3. Drops trips dated outside their file's month (the TLC files contain a few) and zones 264/265 ("Unknown",
   "Outside of NYC").
4. Counts pickups per zone and hour, then writes a **complete grid**: every zone, every hour, with 0 when there
   was no pickup. The lag features can then shift by rows without skipping hours.

Measured (`reports/data_summary.json`): **1,148,784 rows** = 263 zones × 4,368 hours (1 January to 30 June 2020),
**16,724,804 pickups**.

| Month | Pickups |
|---|---|
| January 2020 | 6,357,680 |
| February 2020 | 6,255,661 |
| March 2020 | 2,985,979 |
| April 2020 | 235,738 |
| May 2020 | 344,995 |
| June 2020 | 544,751 |

## Task and splits

Predict the number of pickups in a zone for the next hour from the recent hours of that zone (lags of 1, 2, 3, 24
and 168 hours, the mean of the last 24 hours) and the calendar (hour of day, day of week, weekend).

**Split by time, never at random** (`params.yaml`, `periods`):

| Period | Days | Role |
|---|---|---|
| Train | 8 January to 15 February 2020 | fit the reference model (starts on the 8th so the 168-hour lag exists) |
| Validation | 16 to 29 February 2020 | score it; this is also the **reference** the drift monitor compares against |
| Drift | March to June 2020 | COVID-19: demand collapses from mid-March |

Weekly pickups around the break (from `data/processed/zone_hour.parquet`):

| Week | Pickups |
|---|---|
| 24 Feb to 1 Mar | 1,561,968 |
| 2 to 8 Mar | 1,497,978 |
| 9 to 15 Mar | 990,807 (−34%) |
| 16 to 22 Mar | 228,366 (−77%) |
| 23 to 29 Mar | 71,644 (−69%) |

On top of the real drift, the incident report injects a fault of our own: three busy zones go silent for a week
(see [incident_report.md](incident_report.md)).
