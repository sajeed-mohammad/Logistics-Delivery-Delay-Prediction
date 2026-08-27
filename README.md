# Logistics-Delivery-Delay-Prediction
#  Logistics Delivery Performance & Delay Prediction

##  Project Overview

This project focuses on analyzing logistics delivery performance and predicting whether a delivery will be delayed or completed on time.

The project combines Exploratory Data Analysis (EDA), data visualization, Machine Learning, feature importance analysis, and business insights to identify the major factors associated with delivery delays.

A **Random Forest Classifier** is used to predict delivery delays based on operational and delivery-related features.

> **Dataset Note:** The dataset used in this project is synthetic and was created for educational and portfolio purposes.

---

##  Objectives

- Analyze historical logistics delivery performance.
- Identify factors associated with delivery delays.
- Calculate on-time delivery and delay rates.
- Analyze the impact of traffic and weather conditions.
- Analyze transportation and warehouse performance.
- Build a Machine Learning model for delivery-delay prediction.
- Evaluate the model using standard classification metrics.
- Identify the most important predictive features.
- Generate actionable business recommendations.

---

##  Dataset

The dataset contains **1,000 delivery records and 13 columns**.

### Main Features

| Feature | Description |
|---|---|
| Order_ID | Unique delivery identifier |
| Warehouse | Warehouse associated with the delivery |
| Customer_Location | Customer location category |
| Delivery_Distance_km | Delivery distance in kilometers |
| Transportation_Mode | Transportation method |
| Package_Weight_kg | Package weight in kilograms |
| Delivery_Priority | Delivery priority |
| Weather_Condition | Weather condition |
| Traffic_Level | Traffic condition |
| Warehouse_Processing_Time_min | Warehouse processing time |
| Planned_Delivery_Time_hr | Planned delivery time |
| Actual_Delivery_Time_hr | Actual delivery time |
| Delivery_Delay | Target variable indicating delivery delay |

### Target Variable

```text
0 → On Time
1 → Delayed
