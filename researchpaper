# Factory Reallocation and Shipping Optimization Recommendation System for Nassau Candy Distributor

## Abstract

Nassau Candy Distributor currently assigns products to factories using static rules, which can create inefficient shipping routes, long lead times, and avoidable logistics costs. This project develops a Streamlit-based decision intelligence system that predicts shipping lead time, simulates alternate factory assignments, and recommends factory-product reallocations that balance operational speed with profit stability. Using 9,994 order records across 15 products and 4 customer regions, the system evaluates multiple machine learning models and ranks reassignment scenarios using predicted lead-time reduction, profit stability, scenario confidence, and operational risk.

## Problem Statement

The organization needs a scalable way to answer: **which products should be reassigned to different factories to improve shipping performance without reducing profitability?** Existing factory assignment rules do not simulate alternate configurations or quantify the business impact before implementation. The proposed system solves this by combining predictive modeling, route analysis, and recommendation logic in an interactive dashboard.

## Dataset Description

The dataset contains historical order, product, customer, shipping, sales, cost, and gross profit information. Key fields include order date, ship date, ship mode, destination city/state/region, product division, product name, sales, units, gross profit, and manufacturing cost. Factory coordinates and product-factory mappings were added from the project specification.

Summary:

| Metric | Value |
|---|---:|
| Records analyzed | 9,994 |
| Unique orders | 8,389 |
| Products | 15 |
| Customer regions | 4 |
| Average calculated lead time | 1,321.46 days |

## Methodology

First, order and ship dates were converted into a calculated lead-time target. Product names were mapped to their current factories, and each destination state was approximated using state centroid coordinates. Factory-to-destination distance was calculated using the Haversine formula.

Next, categorical features such as product, division, region, ship mode, and factory were one-hot encoded. Numerical features including distance, units, sales, gross profit, margin rate, month, and weekday were standardized. Extreme lead-time outliers were filtered before model training.

Three regression models were evaluated:

| Model | RMSE | MAE | R2 |
|---|---:|---:|---:|
| Random Forest | 243.43 | 198.80 | 0.123 |
| Gradient Boosting | 251.52 | 205.82 | 0.064 |
| Linear Regression | 257.32 | 205.34 | 0.020 |

Random Forest was selected as the best-performing model because it produced the lowest RMSE and MAE.

## Recommendation Logic

For each product, the system simulates assignment to every available factory:

- Lot's O' Nuts
- Wicked Choccy's
- Sugar Shack
- Secret Factory
- The Other Factory

Each scenario is scored using predicted lead time, percentage lead-time reduction, profit stability, scenario confidence, and risk. The recommendation score prioritizes faster delivery while protecting gross margin.

## Key Results

Top recommended reallocations from the simulation engine:

| Product | Current Factory | Recommended Factory | Lead-Time Reduction | Profit Stability | Risk |
|---|---|---|---:|---:|---|
| Nerds | Sugar Shack | Secret Factory | 9.16% | 100.0% | Medium |
| Laffy Taffy | Sugar Shack | Secret Factory | 7.68% | 100.0% | Medium |
| SweeTARTS | Sugar Shack | Secret Factory | 7.58% | 100.0% | Medium |
| Wonka Gum | Secret Factory | Wicked Choccy's | 3.09% | 100.0% | Low |
| Wonka Bar - Milk Chocolate | Wicked Choccy's | Lot's O' Nuts | 1.49% | 100.0% | Medium |

The results show that several sugar products currently assigned to Sugar Shack may benefit from reassignment to Secret Factory under modeled conditions. Wonka Gum also shows a lower-risk opportunity for reassignment to Wicked Choccy's.

## Streamlit Application

The final dashboard includes:

- Factory Optimization Simulator
- What-if Scenario Analysis
- Recommendation Dashboard
- Route Clustering
- Risk and Profit Impact Panel
- Downloadable recommendation CSV

Users can select a product, destination region, ship mode, and speed-versus-profit priority to compare current and recommended factory assignments.

## Business Impact

The system helps leadership move from descriptive reporting to decision intelligence. Instead of relying only on static factory rules, Nassau Candy can test operational scenarios before execution, identify slow routes, and prioritize reallocation decisions that improve shipping efficiency while maintaining profit stability.

## Conclusion

This project demonstrates how predictive analytics and optimization logic can support factory reallocation decisions for Nassau Candy Distributor. The Random Forest model provides the strongest predictive baseline, while the scenario engine converts predictions into actionable recommendations. The Streamlit dashboard makes the analysis usable for business stakeholders by providing interactive simulation, ranked recommendations, and risk visibility.
