# Ford Line Health Monitor

AI solution 09 of 14 · Operations · built for **Ford Motor Company**

Prepared by **K Mohan Kumar**.

**Live website:** https://mohankumar3548.github.io/ford-predictive-maintenance-ai/

## Problem statement

Unexpected equipment failures are causing production downtime and increasing maintenance costs. Develop an AI system that analyzes machine sensor data, maintenance history and production schedules to identify failure patterns and recommend preventive maintenance actions.

## Why Ford Motor Company

A Ford assembly plant depends on presses, welding robots and conveyors, and one unplanned breakdown can stop the whole line. This tool reads machine sensors, predicts which machine fails next and books the repair into planned downtime.

## Project structure

```
ford-predictive-maintenance-ai/
├── index.html                 Website entry point (served by GitHub Pages)
├── frontend/
│   ├── css/style.css          Layout, typography, light and dark themes
│   └── js/
│       ├── engine.js          Maths, models, charts, page layout and AI analyst panel
│       └── model.js           Sample data and the algorithm for this problem (browser version)
├── app.py                     Streamlit dashboard (Python version of the website)
├── src/
│   ├── data_generator.py      Builds the sample dataset and saves it to data/
│   ├── model.py               The algorithm in Python (pandas, NumPy, scikit-learn)
│   ├── report.py              Writes results to reports/
│   ├── utils.py               Formatting and forecasting helpers
│   └── meta.json              Problem statement, company and algorithm steps
├── data/                      Sample data as CSV
├── reports/                   Results: results.json, summary.md and CSV tables
├── tests/test_model.py        pytest tests for the data and the model
├── requirements.txt
└── pytest.ini
```

## Algorithm

1. **Collect sensor and maintenance data**: Vibration, temperature, motor current, work orders, schedule
2. **Build health features**: Each sensor vs the machine's own healthy baseline (z-score)
3. **Anomaly detection**: Health index from combined z-scores, flag above 3
4. **Remaining useful life**: Trend of the health index to the failure level
5. **7-day failure probability**: Logistic model trained on past failure windows
6. **Failure patterns**: K-means on past failure signatures
7. **Schedule maintenance**: Book each job into the last planned window before failure
8. **Work orders**: AI analyst drafts technician instructions

The same algorithm is implemented twice: in JavaScript (`frontend/js/model.js`) so the website runs in any browser with no server, and in Python (`src/model.py`) for analysis, the Streamlit dashboard and the tests.

## How to run

**Website:** open the live link above, or open `index.html` in a browser.

**Python:**

```bash
pip install -r requirements.txt
python -m src.data_generator     # create the sample data in data/
python -m src.model              # run the algorithm and write reports/
streamlit run app.py             # open the interactive dashboard
pytest                           # run the tests
```

## Data

24 machines on one body-and-assembly line, 120 days of daily sensor readings and 3 years of failure history (sample data). Files: `data/failure_history.csv`, `data/machines.csv`, `data/sensor_readings.csv`. All figures are illustrative and are not Ford Motor Company's actual numbers.

## Results (default settings, Python run)

| Measure | Value | Note |
|---|---|---|
| Machines at risk | 3 | of 24 on the line |
| Highest 7-day risk | 99% | STAM-03, Stamping press |
| Downtime avoided | 51 hrs | Unplanned stops turned into planned work |
| Saving | $6.1M | At $120K per downtime hour |

### Recommended actions

1. **STAM-03: bearing or alignment wear.** Stamping press at 99% risk of failing within 7 days. No planned window comes early enough; stop at the next shift change.
2. **CONV-05: motor or electrical fault.** Conveyor drive at 38% risk of failing within 7 days. Book the 3-hour job into day 8, midweek gap.
3. **WELD-06: cooling or lubrication fault.** Welding robot at 22% risk of failing within 7 days. Book the 4-hour job into day 19, sunday shutdown.
4. **Target bearing and alignment wear across the line.** 46% of past failures follow this pattern. Add checks for it to every Sunday shutdown.

Full output: [`reports/summary.md`](reports/summary.md) and [`reports/results.json`](reports/results.json).

## AI analyst panel

The website includes an "Ask the AI analyst" panel that answers questions about the results through Claude. It works when the page is opened as a Claude artifact on claude.ai. On GitHub Pages the models, charts and controls run in full and the panel shows that it is unavailable.
