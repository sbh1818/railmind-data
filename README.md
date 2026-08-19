# RailMind Data Pipeline

Automated delay data collection and model retraining for RailMind.

## Workflows

| Workflow | Schedule | What it does |
|----------|----------|-------------|
| Nightly Collector | 12:00 AM IST daily | Fetches station-wise delays for 200 trains |
| Biweekly Retrainer | 3:00 AM IST, 1st & 15th | Retrains CatBoost, updates lookup table |

## Structure

```
data/
├── historical/daily_delays_flat.csv  (1M historical records)
├── daily_collected/delays_YYYY-MM-DD.csv (nightly collections)
└── top200_trains.json
models/
├── delay_regressor_final.pkl
├── delay_classifier_final.pkl
├── delay_lookup_table.csv
└── final_model_metadata.json
scripts/
├── collect_delays.py
└── retrain_model.py
```

## Manual Trigger

Go to Actions tab → select workflow → "Run workflow" button.
