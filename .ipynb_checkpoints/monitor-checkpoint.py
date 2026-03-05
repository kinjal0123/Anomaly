import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import config

# --- PART 1: DATA GENERATION & API FETCH ---
def get_combined_data():
    np.random.seed(42)
    rows_per_keyword = 100 
    dates = pd.date_range(end=pd.Timestamp.today(), periods=rows_per_keyword)
    rows = []

    for keyword in config.KEYWORDS:
        base_impressions = np.random.randint(1000, 5000)
        inventory = np.random.randint(20, 100)
        for i in range(rows_per_keyword):
            trend = 1 + (i * 0.001)
            impressions = int(base_impressions * trend * np.random.uniform(0.9, 1.1))
            ctr = np.random.uniform(0.02, 0.08)
            clicks = int(impressions * ctr)
            spend = clicks * np.random.uniform(5, 15)
            orders = int(clicks * np.random.uniform(0.1, 0.3))
            sales = orders * np.random.uniform(300, 800)
            bid = (spend/clicks) * np.random.uniform(0.9, 1.2) if clicks > 0 else 0
            inventory = max(0, inventory - orders)
            
            rows.append([dates[i], keyword, impressions, clicks, spend, sales, orders, inventory, bid])

    df = pd.DataFrame(rows, columns=["date","keyword","impressions","clicks","spend","sales","orders","inventory_level","bid"])
    
    # Calculate Metrics
    df['acos'] = df['spend'] / df['sales'].replace(0, np.nan)
    df['cvr'] = df['orders'] / df['clicks'].replace(0, np.nan)
    df['sales_change_1d'] = df.groupby('keyword')['sales'].pct_change()
    df = df.fillna(0)
    
    return df

# --- PART 2: ANOMALY DETECTION ---
def detect_anomalies(df):
    features = ['acos', 'cvr', 'sales_change_1d', 'inventory_level']
    df_model = df[features].fillna(0)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_model)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly_signal'] = model.fit_predict(scaled_data)
    
    # Sirf aaj ki date ki anomalies filter karein
    latest_date = df['date'].max()
    return df[(df['date'] == latest_date) & (df['anomaly_signal'] == -1)]

# --- PART 3: REASONING & ALERTING ---
def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, data=data)

def process_and_alert():
    df = get_combined_data()
    anomalies = detect_anomalies(df)
    
    
    if anomalies.empty:
        print("No anomalies today. Everything is fine!")
        return

    for _, row in anomalies.iterrows():
        reasons = []
        if row['sales_change_1d'] < -0.3: reasons.append(f"🔻 Sales dropped {abs(row['sales_change_1d']):.1%}")
        if row['inventory_level'] < config.THRESHOLD_INV: reasons.append(f"📦 Low Inventory ({row['inventory_level']})")
        if row['acos'] > config.THRESHOLD_ACOS: reasons.append(f"💸 High ACOS ({row['acos']:.1%})")
        if row['sales_change_1d'] > 0.5: reasons.append(f"🚀 Sales Surge detected!")

        reason_str = " | ".join(reasons) if reasons else "Unusual pattern in metrics"
        
        alert_msg = (
            f"⚠️ *AMAZON BUSINESS ALERT*\n\n"
            f"Product: *{row['keyword']}*\n"
            f"Status: {reason_str}\n\n"
            f"Sales: ₹{row['sales']:.0f}\n"
            f"ACOS: {row['acos']:.1%}\n"
            f"Stock: {row['inventory_level']}"
        )
        send_telegram_msg(alert_msg)
        print(f"Alert sent for {row['keyword']}")

if __name__ == "__main__":
    process_and_alert()