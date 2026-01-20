# ==============================================================================
# UIDAI HACKATHON 2026 - COMPLETE ANALYTICS PIPELINE FOR GOOGLE COLAB
# Dataset: Mar-Dec 2025 (10-month enrollment snapshot)
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Professional styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

print("="*80)
print("   UIDAI ENROLLMENT ANALYTICS - COMPREHENSIVE INSIGHTS")
print("="*80)

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================
print("\n[STEP 1/8] DATA LOADING & PREPROCESSING")
print("-" * 80)

try:
    df1 = pd.read_csv('enrolment_part1.csv')
    df2 = pd.read_csv('enrolment_part2.csv')
    df3 = pd.read_csv('enrolment_part3.csv')
    enrolment_df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"✓ Loaded {enrolment_df.shape[0]:,} records")
except FileNotFoundError:
    print("⚠️ ERROR: Upload CSV files to Colab first!")
    raise

# Date parsing
enrolment_df['date'] = pd.to_datetime(enrolment_df['date'], dayfirst=True, errors='coerce')
enrolment_df = enrolment_df.dropna(subset=['date'])

# Temporal features
enrolment_df['year'] = enrolment_df['date'].dt.year
enrolment_df['month'] = enrolment_df['date'].dt.month
enrolment_df['month_name'] = enrolment_df['date'].dt.strftime('%B')
enrolment_df['weekday'] = enrolment_df['date'].dt.day_name()

# Standardize states
enrolment_df['state'] = enrolment_df['state'].astype(str).str.strip().str.title()
enrolment_df['state'] = enrolment_df['state'].str.replace(r'\s+', ' ', regex=True)
enrolment_df = enrolment_df[~enrolment_df['state'].str.isnumeric()]

state_mapping = {
    'West Bangal': 'West Bengal', 'Westbengal': 'West Bengal',
    'Orissa': 'Odisha', 'Pondicherry': 'Puducherry',
    'Andaman & Nicobar Islands': 'Andaman And Nicobar Islands',
    'Dadra & Nagar Haveli': 'Dadra And Nagar Haveli And Daman And Diu',
    'Daman & Diu': 'Dadra And Nagar Haveli And Daman And Diu'
}
enrolment_df['state'] = enrolment_df['state'].replace(state_mapping)

# Total enrollments
enrolment_df['total_enrolments'] = (enrolment_df['age_0_5'] + 
                                     enrolment_df['age_5_17'] + 
                                     enrolment_df['age_18_greater'])

print(f"✓ Date range: {enrolment_df['date'].min().date()} to {enrolment_df['date'].max().date()}")
print(f"✓ States: {enrolment_df['state'].nunique()} | Districts: {enrolment_df['district'].nunique()}")
print(f"✓ Total enrollments: {enrolment_df['total_enrolments'].sum():,}")

# ==============================================================================
# INSIGHT 1: AGE COHORT DISTRIBUTION (UNIVARIATE)
# ==============================================================================
print("\n[STEP 2/8] INSIGHT 1: AGE COHORT SNAPSHOT")
print("-" * 80)

age_totals = enrolment_df[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
age_pct = (age_totals / age_totals.sum() * 100)

print("\n📊 Age Distribution:")
print(f"   0-5 years:  {age_totals['age_0_5']:>10,} ({age_pct['age_0_5']:.1f}%)")
print(f"   5-17 years: {age_totals['age_5_17']:>10,} ({age_pct['age_5_17']:.1f}%)")
print(f"   18+ years:  {age_totals['age_18_greater']:>10,} ({age_pct['age_18_greater']:.1f}%)")

lifecycle_gap = ((age_totals['age_0_5'] - age_totals['age_5_17']) / age_totals['age_0_5'] * 100)
print(f"\n🔍 FINDING: {lifecycle_gap:.1f}% gap → ~{int(age_totals['age_0_5'] - age_totals['age_5_17']):,} MBU backlog")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#3498db', '#e74c3c', '#2ecc71']
bars = ax1.bar(['0-5 Years', '5-17 Years', '18+ Years'], age_totals, color=colors, edgecolor='black')
ax1.set_title('Age Cohort Distribution', fontweight='bold', fontsize=14)
ax1.set_ylabel('Total Enrollments', fontweight='bold')
for bar, val, pct in zip(bars, age_totals, age_pct):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, 
             f'{int(val):,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

ax2.pie(age_totals, labels=['0-5 Years', '5-17 Years', '18+ Years'],
        autopct='%1.1f%%', colors=colors, startangle=90,
        textprops={'fontweight': 'bold'})
ax2.set_title('Proportional Distribution', fontweight='bold', fontsize=14)

plt.tight_layout()
plt.savefig('insight_1_age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# INSIGHT 2: MONTHLY TRENDS (BIVARIATE)
# ==============================================================================
print("\n[STEP 3/8] INSIGHT 2: MONTHLY MOMENTUM")
print("-" * 80)

month_order = ['March', 'April', 'May', 'June', 'July', 'August', 
               'September', 'October', 'November', 'December']

monthly_data = enrolment_df.groupby('month_name')['total_enrolments'].sum().reindex(month_order)
monthly_data_df = pd.DataFrame(monthly_data)
monthly_data_df['mom_growth'] = monthly_data_df['total_enrolments'].pct_change() * 100

print("\n📈 Monthly Performance:")
# Replace lines 127-135 with this fixed version:

print("\n📈 Monthly Performance:")
for month in month_order:
    if month in monthly_data_df.index:
        total = monthly_data_df.loc[month, 'total_enrolments']
        growth = monthly_data_df.loc[month, 'mom_growth']
        # Check if values are valid (not NaN)
        if pd.notna(total):
            if pd.notna(growth):
                print(f"   {month:10s}: {int(total):>10,} ({growth:>+6.1f}% MoM)")
            else:
                print(f"   {month:10s}: {int(total):>10,} (baseline)")
    # Skip months not in data (don't print anything)

# Also update peak_month calculation to only use available months:
available_months = monthly_data.dropna()
if len(available_months) > 0:
    peak_month = available_months.idxmax()
    print(f"\n🔍 FINDING: Peak = {peak_month} ({int(available_months[peak_month]):,})")
else:
    print(f"\n🔍 FINDING: No valid monthly data found")
peak_month = monthly_data.idxmax()
print(f"\n🔍 FINDING: Peak = {peak_month} ({int(monthly_data[peak_month]):,})")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x_pos = range(len(month_order))
axes[0].plot(x_pos, monthly_data.values, marker='o', linewidth=2.5, markersize=8, color='#2c3e50')
axes[0].fill_between(x_pos, monthly_data.values, alpha=0.3, color='#3498db')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(month_order, rotation=45, ha='right')
axes[0].set_title('Monthly Enrollment Trend', fontweight='bold', fontsize=14)
axes[0].set_ylabel('Total Enrollments', fontweight='bold')
axes[0].grid(alpha=0.3)

growth_vals = monthly_data_df['mom_growth'].fillna(0).values
colors_growth = ['#2ecc71' if x > 0 else '#e74c3c' for x in growth_vals]
axes[1].bar(x_pos, growth_vals, color=colors_growth, edgecolor='black')
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(month_order, rotation=45, ha='right')
axes[1].set_title('Month-over-Month Growth', fontweight='bold', fontsize=14)
axes[1].set_ylabel('Growth Rate (%)', fontweight='bold')

plt.tight_layout()
plt.savefig('insight_2_monthly_trends.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# INSIGHT 3: GEOGRAPHIC CONCENTRATION (BIVARIATE)
# ==============================================================================
print("\n[STEP 4/8] INSIGHT 3: GEOGRAPHIC DISTRIBUTION")
print("-" * 80)

state_totals = enrolment_df.groupby('state')['total_enrolments'].sum().sort_values(ascending=False)
top_5 = state_totals.head(5)
bottom_5 = state_totals.tail(5)
top_5_pct = (top_5.sum() / state_totals.sum() * 100)

print(f"\n🏆 TOP 5 STATES:")
for state, count in top_5.items():
    print(f"   {state:35s}: {int(count):,}")

print(f"\n⚠️ BOTTOM 5 STATES:")
for state, count in bottom_5.items():
    print(f"   {state:35s}: {int(count):,}")

print(f"\n🔍 FINDING: Top 5 control {top_5_pct:.1f}% of enrollments")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.barh(range(len(top_5)), top_5.values, color='#2ecc71', edgecolor='black')
ax1.set_yticks(range(len(top_5)))
ax1.set_yticklabels(top_5.index)
ax1.set_xlabel('Total Enrollments', fontweight='bold')
ax1.set_title('🏆 Top 5 States', fontweight='bold', fontsize=14)
ax1.invert_yaxis()

ax2.barh(range(len(bottom_5)), bottom_5.values, color='#e74c3c', edgecolor='black')
ax2.set_yticks(range(len(bottom_5)))
ax2.set_yticklabels(bottom_5.index)
ax2.set_xlabel('Total Enrollments', fontweight='bold')
ax2.set_title('⚠️ Bottom 5 States', fontweight='bold', fontsize=14)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('insight_3_geographic.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# INSIGHT 4: DISTRICT INEQUALITY (TRIVARIATE)
# ==============================================================================
print("\n[STEP 5/8] INSIGHT 4: DISTRICT-LEVEL INEQUALITY")
print("-" * 80)

district_data = enrolment_df.groupby(['state', 'district'])['total_enrolments'].sum().reset_index()

state_cv = district_data.groupby('state')['total_enrolments'].agg([
    ('mean', 'mean'),
    ('std', 'std'),
    ('count', 'count')
])
state_cv['cv'] = (state_cv['std'] / state_cv['mean'] * 100)
state_cv = state_cv[state_cv['count'] >= 3]
state_cv = state_cv.sort_values('cv', ascending=False)

print(f"\n📊 Top 10 States with Uneven District Coverage:")
for state, row in state_cv.head(10).iterrows():
    print(f"   {state:35s}: CV = {row['cv']:>6.1f}%")

print(f"\n🔍 FINDING: High CV = uneven district penetration within state")

# Visualization
plt.figure(figsize=(12, 8))
top_15_cv = state_cv.head(15)
colors_cv = ['#e74c3c' if x > 150 else '#f39c12' if x > 100 else '#2ecc71' for x in top_15_cv['cv']]
plt.barh(range(len(top_15_cv)), top_15_cv['cv'], color=colors_cv, edgecolor='black')
plt.yticks(range(len(top_15_cv)), top_15_cv.index)
plt.axvline(100, color='orange', linestyle='--', linewidth=2, label='Moderate')
plt.axvline(150, color='red', linestyle='--', linewidth=2, label='High')
plt.xlabel('Coefficient of Variation (%)', fontweight='bold')
plt.title('District-Level Inequality Index', fontweight='bold', fontsize=14)
plt.gca().invert_yaxis()
plt.legend()
plt.tight_layout()
plt.savefig('insight_4_district_inequality.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# INSIGHT 5: ANOMALY DETECTION (ADVANCED ANALYTICS)
# ==============================================================================
print("\n[STEP 6/8] INSIGHT 5: STATISTICAL ANOMALY DETECTION")
print("-" * 80)

pin_data = enrolment_df.groupby('pincode')[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
pin_data['total'] = pin_data.sum(axis=1)
pin_data = pin_data[pin_data['total'] > 50]

pin_data['child_pct'] = pin_data['age_0_5'] / pin_data['total'] * 100
pin_data['adult_pct'] = pin_data['age_18_greater'] / pin_data['total'] * 100

# Z-score anomaly detection
pin_data['child_zscore'] = np.abs(stats.zscore(pin_data['child_pct']))
pin_data['adult_zscore'] = np.abs(stats.zscore(pin_data['adult_pct']))

anomalies = pin_data[(pin_data['child_zscore'] > 3) | (pin_data['adult_zscore'] > 3)]

print(f"\n🚨 Anomaly Detection Results:")
print(f"   Total centers: {len(pin_data):,}")
print(f"   Flagged: {len(anomalies)} ({len(anomalies)/len(pin_data)*100:.2f}%)")
print(f"\n🔍 FINDING: {len(anomalies)} centers need audit")

# Visualization
plt.figure(figsize=(10, 7))
plt.scatter(pin_data['child_pct'], pin_data['adult_pct'], 
            alpha=0.4, s=40, label='Normal', color='steelblue')
plt.scatter(anomalies['child_pct'], anomalies['adult_pct'], 
            color='red', s=100, label=f'Anomalies (n={len(anomalies)})', 
            edgecolors='black', marker='X', zorder=5)
plt.xlabel('Child Enrollment %', fontweight='bold')
plt.ylabel('Adult Enrollment %', fontweight='bold')
plt.title('Anomaly Detection (3-Sigma Method)', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('insight_5_anomalies.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# INSIGHT 6: WEEKDAY EFFICIENCY (OPERATIONAL)
# ==============================================================================
print("\n[STEP 7/8] INSIGHT 6: WEEKDAY OPERATIONAL EFFICIENCY")
print("-" * 80)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_counts = enrolment_df.groupby('weekday')['total_enrolments'].sum().reindex(day_order)
avg_daily = weekday_counts.mean()
efficiency = (weekday_counts / avg_daily * 100)

weekend_util = (weekday_counts[['Saturday', 'Sunday']].sum() / weekday_counts.sum() * 100)

print(f"\n📊 Daily Performance:")
for day in day_order:
    count = weekday_counts[day]
    eff = efficiency[day]
    status = "🟢" if eff > 110 else "🔴" if eff < 90 else "🟡"
    print(f"   {status} {day:10s}: {int(count):>10,} ({eff:.1f}%)")

print(f"\n🔍 FINDING: Weekend utilization = {weekend_util:.1f}%")

# Visualization
plt.figure(figsize=(12, 6))
colors_week = ['#2ecc71' if e > 110 else '#e74c3c' if e < 90 else '#f39c12' for e in efficiency]
bars = plt.bar(day_order, weekday_counts, color=colors_week, edgecolor='black')
plt.axhline(avg_daily, color='black', linestyle='--', label='Average', linewidth=2)
plt.ylabel('Total Enrollments', fontweight='bold')
plt.title('Weekday Efficiency Pattern', fontweight='bold', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.legend()
for bar, val in zip(bars, weekday_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height, 
             f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig('insight_6_weekday.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# BONUS INSIGHT 7: PREDICTIVE FORECASTING (ML)
# ==============================================================================
print("\n[STEP 8/8] BONUS INSIGHT 7: ENROLLMENT FORECASTING")
print("-" * 80)

monthly_ts = enrolment_df.groupby(['year', 'month'])['total_enrolments'].sum().reset_index()
monthly_ts = monthly_ts.sort_values(['year', 'month'])
monthly_ts['month_index'] = range(len(monthly_ts))

X = monthly_ts['month_index'].values.reshape(-1, 1)
y = monthly_ts['total_enrolments'].values
model = LinearRegression()
model.fit(X, y)

# Forecast next 3 months
future_months = np.array([len(monthly_ts), len(monthly_ts)+1, len(monthly_ts)+2]).reshape(-1, 1)
forecast = model.predict(future_months)
r_squared = model.score(X, y)

print(f"\n🔮 Forecast Model:")
print(f"   R² Score: {r_squared:.3f}")
print(f"\n📈 Next 3 Months Forecast:")
forecast_names = ['Jan 2026', 'Feb 2026', 'Mar 2026']
for name, pred in zip(forecast_names, forecast):
    print(f"   {name}: ~{int(pred):,} enrollments")

print(f"\n🔍 FINDING: {'Upward' if model.coef_[0] > 0 else 'Downward'} trend detected")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(monthly_ts['month_index'], monthly_ts['total_enrolments'], 
         marker='o', linewidth=2, label='Actual', color='#2c3e50')
plt.plot(monthly_ts['month_index'], model.predict(X), 
         linestyle='--', linewidth=2, label='Trend', color='#e74c3c')
plt.plot(future_months, forecast, 
         marker='s', markersize=10, linestyle='--', linewidth=2, 
         label='Forecast', color='#27ae60')
plt.fill_between(future_months.flatten(), forecast * 0.9, forecast * 1.1, 
                 alpha=0.2, color='#27ae60')
plt.xlabel('Month Index', fontweight='bold')
plt.ylabel('Total Enrollments', fontweight='bold')
plt.title('Enrollment Trend & Q1 2026 Forecast', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('insight_7_forecast.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# BONUS INSIGHT 8: DISTRICT CLUSTERING (ML)
# ==============================================================================
print("\n[BONUS] INSIGHT 8: DISTRICT PERFORMANCE CLUSTERING")
print("-" * 80)

district_features = enrolment_df.groupby('district').agg({
    'total_enrolments': 'sum',
    'age_0_5': 'sum',
    'age_5_17': 'sum',
    'age_18_greater': 'sum'
}).reset_index()

district_features['child_ratio'] = district_features['age_0_5'] / district_features['total_enrolments']
district_features = district_features[district_features['total_enrolments'] > 500]

X_cluster = district_features[['total_enrolments', 'child_ratio']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
district_features['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n📊 District Segmentation ({len(district_features)} districts):")
for cluster_id in range(4):
    cluster_data = district_features[district_features['cluster'] == cluster_id]
    avg_enrol = cluster_data['total_enrolments'].mean()
    count = len(cluster_data)
    print(f"   Cluster {cluster_id}: {count} districts | Avg: {int(avg_enrol):,}")

print(f"\n🔍 FINDING: 4 distinct performance segments identified")

# Visualization
plt.figure(figsize=(10, 7))
scatter = plt.scatter(district_features['total_enrolments'], 
                     district_features['child_ratio'] * 100,
                     c=district_features['cluster'], 
                     cmap='viridis', s=80, alpha=0.6, edgecolors='black')
plt.xlabel('Total Enrollments (log scale)', fontweight='bold')
plt.ylabel('Child Enrollment %', fontweight='bold')
plt.xscale('log')
plt.title('District Clustering by Performance', fontweight='bold', fontsize=14)
plt.colorbar(scatter, label='Cluster ID')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('insight_8_clustering.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("🏆 EXECUTIVE SUMMARY")
print("="*80)

print(f"""
📊 DATASET OVERVIEW:
   • Total Records: {enrolment_df.shape[0]:,}
   • Date Range: Mar-Dec 2025 (10 months)
   • States: {enrolment_df['state'].nunique()} | Districts: {enrolment_df['district'].nunique()}
   • Total Enrollments: {enrolment_df['total_enrolments'].sum():,}

🔍 KEY FINDINGS:

1. LIFECYCLE GAP (Univariate)
   → {lifecycle_gap:.1f}% gap between age cohorts
   → MBU backlog: ~{int(age_totals['age_0_5'] - age_totals['age_5_17']):,} children

2. TEMPORAL TRENDS (Bivariate)
   → Peak month: {peak_month}
   → Growth volatility detected

3. GEOGRAPHIC CONCENTRATION (Bivariate)
   → Top 5 states: {top_5_pct:.1f}% of enrollments
   → Bottom states need intervention

4. DISTRICT INEQUALITY (Trivariate)
   → {len(state_cv[state_cv['cv'] > 150])} states with high CV
   → Uneven resource distribution

5. ANOMALY DETECTION (Advanced)
   → {len(anomalies)} centers flagged ({len(anomalies)/len(pin_data)*100:.2f}%)
   → Require immediate audit

6. OPERATIONAL EFFICIENCY
   → Weekend utilization: {weekend_util:.1f}%
   → Capacity optimization opportunity

7. PREDICTIVE FORECAST (ML)
   → R² = {r_squared:.3f}
   → Q1 2026 forecast: ~{int(forecast.sum()):,} enrollments

8. STRATEGIC SEGMENTATION (ML)
   → 4 district clusters identified
   → Targeted intervention strategies

📈 IMPACT POTENTIAL:
   • MBU Completion: ~{int((age_totals['age_0_5'] - age_totals['age_5_17']) * 0.3):,} children
   • Geographic Equity: {len(bottom_5)} priority states
   • Quality Assurance: {len(anomalies)} centers for audit
   • Capacity Gain: Weekend optimization opportunity

🎯 RECOMMENDATIONS:
   1. Launch MBU reminder campaigns
   2. Deploy mobile units to bottom states
   3. Audit anomalous centers
   4. Optimize weekend capacity
   5. Implement cluster-based interventions
""")

print("="*80)
print("✓ ANALYSIS COMPLETE - 8 INSIGHTS GENERATED")
print("✓ All visualizations saved as PNG files")
print("="*80)