import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("../data/logistics_data.csv")

print("\nDataset Shape:", df.shape)

# ==========================================
# 2. Check Missing Values
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 3. Check Duplicate Records
# ==========================================

print("\nDuplicate Records:", df.duplicated().sum())

# ==========================================
# 4. Summary Statistics
# ==========================================

print("\nSummary Statistics:")
print(df.describe())

# ==========================================
# 5. Delivery Performance
# ==========================================

on_time_rate = (df["Delivery_Delay"] == 0).mean() * 100
delay_rate = (df["Delivery_Delay"] == 1).mean() * 100

print(f"\nOn-Time Delivery Rate: {on_time_rate:.2f}%")
print(f"Delay Rate: {delay_rate:.2f}%")

# ==========================================
# 6. Delay Analysis
# ==========================================

for column in [
    "Transportation_Mode",
    "Traffic_Level",
    "Weather_Condition",
    "Warehouse"
]:

    print(f"\nDelay Rate by {column}:")

    delay_by_category = (
        df.groupby(column)["Delivery_Delay"].mean() * 100
    )

    print(delay_by_category.round(2))

# ==========================================
# 7. Delay Distribution Graph
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Delivery_Delay"
)

plt.title("Delivery Delay Distribution")
plt.xlabel("Delivery Delay (0 = On Time, 1 = Delayed)")
plt.ylabel("Number of Deliveries")

plt.tight_layout()

plt.savefig(
    "../visualizations/delay_distribution.png"
)

plt.show()

# ==========================================
# 8. Traffic Impact Graph
# ==========================================

traffic_delay = (
    df.groupby("Traffic_Level")["Delivery_Delay"].mean() * 100
)

plt.figure(figsize=(8, 5))

sns.barplot(
    x=traffic_delay.index,
    y=traffic_delay.values
)

plt.title("Delivery Delay Rate by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Delay Rate (%)")

plt.tight_layout()

plt.savefig(
    "../visualizations/traffic_impact.png"
)

plt.show()

# ==========================================
# 9. Distance vs Delivery Time
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Delivery_Distance_km",
    y="Actual_Delivery_Time_hr",
    hue="Delivery_Delay"
)

plt.title("Delivery Distance vs Actual Delivery Time")
plt.xlabel("Delivery Distance (km)")
plt.ylabel("Actual Delivery Time (hours)")

plt.tight_layout()

plt.savefig(
    "../visualizations/distance_vs_delivery.png"
)

plt.show()

# ==========================================
# 10. Correlation Heatmap
# ==========================================

numeric_columns = df.select_dtypes(
    include="number"
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    numeric_columns.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "../visualizations/correlation_heatmap.png"
)

plt.show()

print("\n======================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("Graphs saved in visualizations folder.")
print("======================================")