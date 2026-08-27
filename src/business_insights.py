import pandas as pd

# ==========================================
# BUSINESS INSIGHTS & RECOMMENDATIONS
# ==========================================

df = pd.read_csv("../data/logistics_data.csv")

print("\n======================================")
print("BUSINESS INSIGHTS")
print("======================================")

# ------------------------------------------
# 1. Overall Delivery Performance
# ------------------------------------------

delay_rate = (
    df["Delivery_Delay"].mean() * 100
)

on_time_rate = (
    (df["Delivery_Delay"] == 0).mean() * 100
)

print("\n1. Overall Delivery Performance")
print(f"On-Time Delivery Rate : {on_time_rate:.2f}%")
print(f"Delivery Delay Rate   : {delay_rate:.2f}%")

# ------------------------------------------
# 2. Traffic Impact
# ------------------------------------------

print("\n2. Traffic Impact")

traffic_delay = (
    df.groupby("Traffic_Level")["Delivery_Delay"]
    .mean() * 100
)

print(traffic_delay.round(2))

highest_traffic = traffic_delay.idxmax()

print(
    f"Highest delay rate is observed under "
    f"{highest_traffic} traffic conditions."
)

# ------------------------------------------
# 3. Weather Impact
# ------------------------------------------

print("\n3. Weather Impact")

weather_delay = (
    df.groupby("Weather_Condition")["Delivery_Delay"]
    .mean() * 100
)

print(weather_delay.round(2))

highest_weather = weather_delay.idxmax()

print(
    f"Highest weather-related delay rate is "
    f"observed during {highest_weather} conditions."
)

# ------------------------------------------
# 4. Transportation Impact
# ------------------------------------------

print("\n4. Transportation Mode Impact")

transport_delay = (
    df.groupby("Transportation_Mode")["Delivery_Delay"]
    .mean() * 100
)

print(transport_delay.round(2))

highest_transport = transport_delay.idxmax()

print(
    f"{highest_transport} has the highest delay rate "
    f"among transportation modes."
)

# ------------------------------------------
# 5. Warehouse Performance
# ------------------------------------------

print("\n5. Warehouse Performance")

warehouse_delay = (
    df.groupby("Warehouse")["Delivery_Delay"]
    .mean() * 100
)

print(warehouse_delay.round(2))

highest_warehouse = warehouse_delay.idxmax()

print(
    f"{highest_warehouse} has the highest delivery "
    f"delay rate among warehouses."
)

# ------------------------------------------
# 6. Business Recommendations
# ------------------------------------------

print("\n======================================")
print("BUSINESS RECOMMENDATIONS")
print("======================================")

print("""
1. Traffic-Aware Route Planning
   Use traffic information while planning delivery routes
   and estimating delivery times.

2. Weather-Based Risk Management
   Adjust delivery schedules during severe weather
   conditions and provide additional time buffers.

3. Warehouse Process Optimization
   Identify warehouse bottlenecks and reduce unnecessary
   processing and waiting time.

4. High-Risk Delivery Monitoring
   Use the Random Forest model to identify deliveries
   with a high probability of delay.

5. Transportation Optimization
   Select transportation modes based on distance,
   traffic, priority, and operational conditions.

6. Dynamic Delivery-Time Estimation
   Use distance, traffic, weather, and historical
   delivery information to improve estimated delivery times.

7. Logistics KPI Dashboard
   Monitor delivery performance using KPIs such as
   On-Time Rate, Delay Rate, Average Delivery Time,
   and High-Risk Deliveries.
""")

# ------------------------------------------
# 7. Final Summary
# ------------------------------------------

print("\n======================================")
print("FINAL BUSINESS SUMMARY")
print("======================================")

print("""
The analysis shows that traffic, weather, delivery
distance, warehouse processing time, and transportation
conditions are important factors associated with delivery
performance.

The Random Forest model can help the logistics team
identify potentially delayed deliveries in advance and
take proactive corrective actions.

This can support better route planning, resource
allocation, warehouse operations, and delivery reliability.
""")

print("======================================")
print("BUSINESS ANALYSIS COMPLETED!")
print("======================================")