# Employee Salary Predictor — Streamlit Dashboard (INR)

## Folder contents
```
.
├── app.py                  # the Streamlit dashboard
├── train_model.py          # trains the model + saves model/salary_model.pkl
├── ds_salaries_inr.csv     # synthetic training data, salaries in INR
├── requirements.txt
├── .streamlit/config.toml  # locks the app to a light theme for predictable contrast
└── model/
    └── salary_model.pkl    # trained model + metadata (already generated)
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

The trained model (`model/salary_model.pkl`) is already included, so you can
run the app immediately.

## Features used

- **Years of experience** (numeric)
- **Education** — High School / Bachelor's / Master's / PhD
- Job title, employment type, company size, remote-work ratio

Salary is predicted and shown **only in INR** — the dataset itself is
generated in INR (not converted from USD), so there's no exchange-rate
assumption baked into the numbers.

## Using a real dataset instead of the synthetic one

If you have a real salary dataset with similar columns
(`experience_years`, `education_level`, `employment_type`, `job_title`,
`remote_ratio`, `company_size`, `salary_in_inr`):

1. Save it as `ds_salaries_inr.csv` in this folder (or change `DATA_PATH`
   in `train_model.py`).
2. Run `python train_model.py` to retrain.
3. Run `streamlit run app.py` — the dashboard's dropdowns and slider ranges
   automatically reflect whatever values are in the new data.

## About the CSS / theme

The app pins Streamlit's theme to `light` via `.streamlit/config.toml`.
This matters because Streamlit otherwise follows the viewer's OS/browser
dark-mode setting, which silently flips default text colors — that's what
caused the earlier bug where a widget's text was invisible (white text
landing on a white input box). Every custom-styled block in `app.py` now
also sets its own background *and* text color explicitly, rather than
relying on an inherited default, so it stays legible no matter the
viewer's system theme.
