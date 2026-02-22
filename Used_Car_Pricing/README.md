# What drives the price of a car?

![](images/kurt.jpeg)

Notebook Link: [used_car_pricing.ipynb](used_car_pricing.ipynb)
## Project Overview

This project analyzes a used vehicle dataset to determine the key
factors that influence used car prices.\
The analysis follows the **CRISP-DM (Cross Industry Standard Process for
Data Mining)** framework and culminates in actionable business
recommendations for a used car dealership.

The primary objective is to identify what consumers value most in a used
vehicle and provide inventory optimization guidance for dealers.

------------------------------------------------------------------------
</br>
</br>
</br>
</br>

# CRISP-DM Framework Structure
<img src = images/crisp.png width = 40%/>

## 1. Business Understanding

### Business Objective

Identify the main drivers of used car prices to help dealerships: -
Optimize inventory selection - Improve pricing strategy - Increase sales
velocity - Maximize profit margins

### Data Science Problem

This is framed as a **supervised regression problem**, where:

-   **Target variable:** `price`
-   **Predictors:** vehicle characteristics such as year, manufacturer,
    condition, mileage, fuel type, transmission, etc.
-   **Goal:** Minimize prediction error and interpret feature
    importance.

------------------------------------------------------------------------

## 2. Data Understanding

### Dataset

The dataset (`vehicles.csv`) contains structured information about used
cars including: - Price - Year - Manufacturer - Model - Condition -
Cylinders - Fuel type - Odometer (mileage) - Transmission - Drive type -
Size - Type - Paint color - State

### Key Exploration Steps

-   Shape and structure inspection
-   Data types review
-   Missing value analysis
-   Outlier detection (price, mileage)
-   Distribution analysis
-   Correlation analysis

------------------------------------------------------------------------

## 3. Data Preparation

Major preparation steps include:

-   Removing unrealistic prices (e.g., \$0, extreme outliers)
-   Handling missing values
-   Converting categorical variables using one-hot encoding
-   Log transformation of price (to reduce skew)
-   Feature scaling for numeric variables
-   Train/test split
-   Pipeline construction using `sklearn`

------------------------------------------------------------------------

## 4. Modeling

Multiple regression models were built and cross-validated:

-   Linear Regression
-   Ridge Regression
-   Lasso Regression
-   Random Forest Regressor

Model performance evaluated using: - RMSE (Root Mean Squared Error) - R²
Score - Cross-validation (5-fold)

Hyperparameters were tuned where appropriate.

------------------------------------------------------------------------

## 5. Evaluation

### Key Insights

Typical strong price drivers: - Vehicle age (newer cars command higher
prices) - Mileage (lower mileage increases price) - Manufacturer and
brand reputation - Vehicle condition - Fuel type (hybrid/electric
premium in some markets) - Transmission type (automatic often preferred)

Tree-based models (e.g., Random Forest) typically provide better
predictive performance due to nonlinear relationships.

------------------------------------------------------------------------

## 6. Deployment -- Business Recommendations

### Inventory Strategy

Dealers should prioritize: - Newer model years - Low mileage vehicles -
Vehicles in "excellent" or "like new" condition - Popular manufacturers
with strong resale value - Automatic transmission vehicles - SUVs and
trucks (if demand supports regional trends)

### Pricing Strategy

-   Use model-based price benchmarks for acquisitions
-   Identify undervalued vehicles using prediction residuals
-   Avoid overpaying for high-mileage or poor-condition vehicles

------------------------------------------------------------------------
</br>
</br>
</br>
</br>


# How to Run the Notebook

1.  Install required packages:

    ``` bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

2.  Ensure `vehicles.csv` is in a sub directory named 'data' from where the notebook is.

3.  Run the notebook sequentially.

------------------------------------------------------------------------

# Repository Structure

-   `used_car_price_crispdm.ipynb` -- Main analysis notebook
-   `vehicles.csv` -- Dataset, in a 'data' sub-directory
-   `README.md` -- Project documentation

------------------------------------------------------------------------

# Business Value Delivered

This project provides:

-   A data-driven pricing framework
-   Clear understanding of price drivers
-   Inventory optimization guidance
-   A repeatable modeling pipeline for future data

------------------------------------------------------------------------
