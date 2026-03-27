"""
DATA v2.2 — Open-Meteo CAMS + ERA5 + Meteostat нақты станция
pip install requests pandas numpy meteostat
python src/data_v2_openmeteo.py
"""
import requests, pandas as pd, numpy as np, os, time
from datetime import datetime

OUT = "data"; os.makedirs(OUT, exist_ok=True)
LAT, LON = 43.25, 76.95
START_STR, END_STR = "2024-01-01", "2025-12-31"
START_DT  = datetime(2024, 1, 1)
END_DT    = datetime(2025, 12, 31, 23)
TIMEZONE  = "Asia/Almaty"
METEOSTAT_WMO = "36870"   # Алматы UAAA, 847 м

def fetch_air_quality():
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {"latitude": LAT, "longitude": LON,
              "start_date": START_STR, "end_date": END_STR,
              "hourly": "pm2_5,pm10,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide,aerosol_optical_depth,dust",
              "timezone": TIMEZONE}
    print("⬇  Open-Meteo Air Quality жүктелуде...")
    h = requests.get(url, params=params, timeout=60).raise_for_status() or requests.get(url, params=params, timeout=60).json()["hourly"]
    # --- fix:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"datetime": pd.to_datetime(h["time"]),
                       "pm25": h["pm2_5"], "pm10": h["pm10"],
                       "no2": h["nitrogen_dioxide"], "o3": h["ozone"],
                       "so2": h["sulphur_dioxide"], "co": h["carbon_monoxide"],
                       "aod": h["aerosol_optical_depth"], "dust": h["dust"]
                       }).set_index("datetime")
    print(f"   ✓ {len(df)} бақылау")
    return df

def fetch_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": LAT, "longitude": LON,
              "start_date": START_STR, "end_date": END_STR,
              "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,surface_pressure,precipitation,cloud_cover,boundary_layer_height",
              "timezone": TIMEZONE, "wind_speed_unit": "ms"}
    print("⬇  Open-Meteo Weather (ERA5) жүктелуде...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({"datetime": pd.to_datetime(h["time"]),
                       "temp": h["temperature_2m"], "humidity": h["relative_humidity_2m"],
                       "wind_spd": h["wind_speed_10m"], "wind_dir": h["wind_direction_10m"],
                       "wind_gust": h["wind_gusts_10m"], "pressure": h["surface_pressure"],
                       "precip": h["precipitation"], "cloud": h["cloud_cover"],
                       "blh": h["boundary_layer_height"]}).set_index("datetime")
    print(f"   ✓ {len(df)} бақылау | {len(df.columns)} метео айнымалы")
    return df

def fetch_meteostat():
    try:
        from meteostat import Stations, Hourly
    except ImportError:
        print("   ⚠  pip install meteostat")
        return pd.DataFrame()
    print(f"⬇  Meteostat жүктелуде (WMO {METEOSTAT_WMO}, UAAA Алматы)...")
    try:
        station = Stations().id("wmo", METEOSTAT_WMO).fetch()
        if station.empty:
            station = Stations().nearby(LAT, LON).fetch().head(1)
        sid = station.index[0]
        df = Hourly(sid, START_DT, END_DT, timezone=TIMEZONE).fetch()
        if df.empty:
            print("   ⚠  Meteostat бос")
            return pd.DataFrame()
        rename = {"temp":"ms_temp","dwpt":"ms_dewpoint","rhum":"ms_humidity",
                  "prcp":"ms_precip","wdir":"ms_wind_dir",
                  "wspd":"ms_wind_spd","wpgt":"ms_wind_gust",
                  "pres":"ms_pressure"}
        df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
        # км/сағ → м/с
        for c in ["ms_wind_spd","ms_wind_gust"]:
            if c in df.columns: df[c] /= 3.6
        ms_cols = [c for c in df.columns if c.startswith("ms_")]
        df = df[ms_cols]
        # толықтық
        fill = df.notna().mean()*100
        print(f"   ✓ {len(df)} сағаттық | толықтық: temp={fill.get('ms_temp',0):.0f}% wind={fill.get('ms_wind_spd',0):.0f}%")
        return df
    except Exception as e:
        print(f"   ✗ {e}")
        return pd.DataFrame()

def engineer_features(df, target):
    out = pd.DataFrame(index=df.index)
    # v1
    for lag in [1,2,3,6,12,24,48,168]:
        out[f"{target}_lag_{lag}"] = df[target].shift(lag)
    for w in [6,24,168]:
        s = df[target].shift(1)
        out[f"{target}_roll_mean_{w}"] = s.rolling(w).mean()
        out[f"{target}_roll_std_{w}"]  = s.rolling(w).std()
        out[f"{target}_roll_max_{w}"]  = s.rolling(w).max()
    out[f"{target}_diff_1"]  = df[target].shift(1) - df[target].shift(2)
    out[f"{target}_diff_24"] = df[target].shift(1) - df[target].shift(25)
    out["hour_sin"]    = np.sin(2*np.pi*df.index.hour/24)
    out["hour_cos"]    = np.cos(2*np.pi*df.index.hour/24)
    out["month_sin"]   = np.sin(2*np.pi*df.index.month/12)
    out["month_cos"]   = np.cos(2*np.pi*df.index.month/12)
    out["dow_sin"]     = np.sin(2*np.pi*df.index.dayofweek/7)
    out["dow_cos"]     = np.cos(2*np.pi*df.index.dayofweek/7)
    out["day_of_year"] = df.index.dayofyear
    # v2 ERA5
    out["wind_u"] = -df["wind_spd"]*np.sin(np.radians(df["wind_dir"]))
    out["wind_v"] = -df["wind_spd"]*np.cos(np.radians(df["wind_dir"]))
    for lag in [1,3,6,24]:
        out[f"temp_lag_{lag}"]     = df["temp"].shift(lag)
        out[f"wind_spd_lag_{lag}"] = df["wind_spd"].shift(lag)
        out[f"humidity_lag_{lag}"] = df["humidity"].shift(lag)
    for w in [6,24]:
        for col in ["temp","wind_spd","humidity","blh"]:
            if col in df: 
                s=df[col].shift(1)
                out[f"{col}_roll_mean_{w}"]=s.rolling(w).mean()
                out[f"{col}_roll_std_{w}"]=s.rolling(w).std()
    out["blh_log"]  = np.log1p(df["blh"].shift(1))
    out["blh_lag_1"] = df["blh"].shift(1)
    out["blh_lag_24"]= df["blh"].shift(24)
    out["inversion_flag"] = ((df["temp"].shift(1)<0)&(df["wind_spd"].shift(1)<2.0)&
                              df.index.month.isin([10,11,12,1,2,3])).astype(float)
    out["calm_flag"]      = (df["wind_spd"].shift(1)<1.5).astype(float)
    out["heating_season"] = df.index.month.isin([10,11,12,1,2,3]).astype(float)
    out["is_winter"]      = df.index.month.isin([12,1,2]).astype(float)
    out["temp_x_wind"]    = df["temp"].shift(1)*df["wind_spd"].shift(1)
    out["precip_lag_1"]   = df["precip"].shift(1)
    out["precip_lag_6"]   = df["precip"].shift(6)
    out["no2_lag_1"]      = df["no2"].shift(1)
    out["aod_lag_1"]      = df["aod"].shift(1)
    out["dust_lag_1"]     = df["dust"].shift(1)
    # v2.2 Meteostat
    for col in [c for c in df.columns if c.startswith("ms_") and c!="ms_weather_code"]:
        for lag in [1,6,24]:
            out[f"{col}_lag_{lag}"] = df[col].shift(lag)
        out[f"{col}_roll_mean_24"] = df[col].shift(1).rolling(24).mean()
    if "ms_temp" in df.columns:
        out["temp_bias_era5_ms"]  = df["temp"].shift(1) - df["ms_temp"].shift(1)
    if "ms_wind_spd" in df.columns:
        out["wind_bias_era5_ms"]  = df["wind_spd"].shift(1) - df["ms_wind_spd"].shift(1)
    # targets
    out["target_1h"]  = df[target].shift(-1)
    out["target_6h"]  = df[target].shift(-6)
    out["target_12h"] = df[target].shift(-12)
    out["target_24h"] = df[target].shift(-24)
    return out

def main():
    print("="*60)
    print("ALMATY AIR QUALITY — DATA v2.2")
    print("Open-Meteo CAMS + ERA5 + Meteostat нақты станция")
    print("="*60)
    aq = fetch_air_quality()
    time.sleep(1)
    wx = fetch_weather()
    print("\n🔗 Open-Meteo біріктірілуде...")
    df = aq.join(wx, how="inner")
    print(f"   ✓ {len(df)} бақылау | {len(df.columns)} айнымалы")
    df.to_csv(f"{OUT}/almaty_raw_v2.csv")
    print(f"   ✓ {OUT}/almaty_raw_v2.csv")

    ms = fetch_meteostat()
    if not ms.empty:
        ms.to_csv(f"{OUT}/almaty_meteostat_hourly.csv")
        print(f"   ✓ {OUT}/almaty_meteostat_hourly.csv")
        # timezone үйлестіру
        if ms.index.tz is None:
            ms.index = ms.index.tz_localize(TIMEZONE)
        if df.index.tz is None:
            df.index = df.index.tz_localize(TIMEZONE)
        df = df.join(ms, how="left")
        n_ms = len([c for c in df.columns if c.startswith("ms_")])
        print(f"   ✓ Meteostat {n_ms} айнымалы қосылды")

    for target in ["pm25", "pm10"]:
        print(f"\n⚙  Белгілер ({target.upper()})...")
        feat = engineer_features(df, target)
        feat = feat.dropna(subset=["target_1h"])
        path = f"{OUT}/features_{target}_v2.csv"
        feat.to_csv(path)
        v1  = len([c for c in feat.columns if f"{target}_lag_" in c or f"{target}_roll_" in c or f"{target}_diff_" in c or c.startswith("hour_") or c.startswith("month_") or c.startswith("dow_")])
        v2e = len([c for c in feat.columns if any(k in c for k in ["temp","wind","humidity","blh","inversion","calm","precip","no2","aod","dust","is_winter","heating","temp_x_wind"]) and "ms_" not in c and "bias" not in c])
        v2m = len([c for c in feat.columns if "ms_" in c or "bias" in c])
        print(f"   ✓ {len(feat)} жол | {len(feat.columns)} белгі")
        print(f"      v1 PM лагтар+уақыт: {v1} | ERA5 метео: {v2e} | Meteostat: {v2m}")
        print(f"   → {path}")

    print("\n✅ ДАЙЫН! Келесі: python src/modeling_v2.py")

if __name__ == "__main__":
    main()
