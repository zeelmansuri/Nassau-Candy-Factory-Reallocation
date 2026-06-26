# Nassau Candy Factory Reallocation & Shipping Optimization

Streamlit decision intelligence project for simulating factory-product reassignment scenarios and ranking shipping optimization recommendations.
Project Link :- https://fbuzuk94khm2zxemxwwirg.streamlit.app
## Features

- Predictive lead-time modeling with Linear Regression, Random Forest, and Gradient Boosting.
- Factory optimization simulator for product, region, and ship-mode scenarios.
- Ranked reassignment recommendations using speed, profit stability, confidence, and risk.
- Route clustering to surface slow or congested region-factory combinations.
- Executive-ready KPI dashboard and downloadable recommendation CSV.

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app expects the dataset at:

```text
data/nassau_candy_orders.csv
```

## Project Structure

```text
app.py
requirements.txt
src/
  optimization.py
data/
  nassau_candy_orders.csv
outputs/
```

## Methodology

1. Parse order and ship dates to calculate lead time.
2. Assign each product to the current factory using the supplied product-factory mapping.
3. Approximate destination coordinates from state centroids and calculate factory-to-market distance.
4. Train and evaluate three regressors using RMSE, MAE, and R2.
5. Simulate alternate factory assignments and rank by lead-time reduction, profit stability, confidence, and risk.
