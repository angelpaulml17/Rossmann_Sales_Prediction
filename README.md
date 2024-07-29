# Rossmann Sales Prediction Project

## Overview

This project aims to predict sales and customer numbers for Rossmann stores using machine learning techniques. The prediction helps in resource management, manpower planning, and identifying the need for further expansion. The dataset includes store-related data, sales, customer numbers, and holiday information. The forecasting accuracy is evaluated using Root Mean Square Percentage Error (RMSPE).

## Project Structure

- **Data Integration**: Merging store data with train and test datasets.
- **Data Cleaning**: Handling missing values, outliers, and categorical data.
- **Feature Engineering**: Creating new features and reducing multicollinearity.
- **Exploratory Data Analysis (EDA)**: Visualizing data to gain insights and guide feature selection.
- **Modeling**: Using XGBoost for prediction due to its efficiency with large datasets and time series data.
- **Evaluation**: Assessing model performance using RMSPE.

## Data Preprocessing

### Data Integration

- Merged store data with train and test datasets on the store variable.

### Data Cleaning

- **Handling Missing Values**:
  - Imputed missing competition distances with 4 times the maximum value.
  - Imputed missing values for competition and promotion columns based on median values grouped by store type and assortment.
- **Outliers**:
  - Removed records with zero sales when stores are open and participating in promotions.
  - Addressed discrepancies in the `Open` variable.
- **Categorical Data**:
  - Encoded categorical variables using LabelEncoder.
  - Mapped `PromoInterval` to numerical values for better trend capture.

### Feature Engineering

- Created `CompOpenSince` and `Promo2Open` to capture the effect of competition and promotion start dates.
- Introduced an interaction variable combining promotion and holiday effects.
- Added state data to account for regional differences in holiday effects.
- Dropped highly correlated features to reduce multicollinearity.

## Exploratory Data Analysis (EDA)

- **Store Type and Assortment**: Significant impact on sales, with more stores leading to higher sales.
- **Promotion**: Increases sales, with the highest impact on day 1.
- **Number of Customers**: Directly proportional to sales.
- **Competition Distance**: Closer competition has a positive impact on sales.
- **Holidays**: School holidays increase sales, while state holidays decrease the number of open stores.
- **Seasonality**: Identified patterns repeating every 2 weeks using visualizations and statistical models.

## Modeling

- **Model Used**: XGBoost Gradient Boosting.
- **Performance**: RMSPE for sales prediction is 0.302 and for customer prediction is 0.293, indicating an average deviation of 30% and 29% from actual values, respectively.

## Results

- Visualizations provided insights into feature importance and relationships.
- The model captured sales patterns effectively, especially with the inclusion of engineered features.

## Conclusion

Effective preprocessing, EDA, and feature engineering are crucial for accurate predictions. The project demonstrates that XGBoost is suitable for handling large datasets and time series data. Future work involves incorporating external factors like weather and using advanced models like RNN, LSTM, SARIMA, and Prophet for improved predictions.

