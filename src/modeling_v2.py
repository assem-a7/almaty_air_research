"""
============================================================
ALMATY AIR QUALITY — MODELING v2.0
NaN қатесі түзетілді: барлық белгілер dropna + imputer
Іске асыру:
    pip install pandas numpy scikit-learn matplotlib
    pip install shap          (қосымша, SHAP визуализация үшін)
    python src/modeling_v2.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("ℹ  SHAP жоқ — pip install shap (қосымша)")

os.makedirs("reports/figures", exist_ok=True)


# ─────────────────────────────────────────────
# МЕТРИКА
# ─────────────────────────────────────────────
def calc_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mask = y_true > 1
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"rmse": round(rmse, 3), "mae": round(mae, 3),
            "r2": round(r2, 3), "mape": round(mape, 1)}


# ─────────────────────────────────────────────
# DIEBOLD-MARIANO ТЕСТ
# ─────────────────────────────────────────────
def dm_test(e1, e2):
    from scipy import stats
    d = e1**2 - e2**2
    n = len(d)
    var_d = np.var(d, ddof=1) / n
    if var_d <= 0:
        return {"stat": 0.0, "p": 1.0, "sig": False}
    dm   = np.mean(d) / np.sqrt(var_d)
    p    = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return {"stat": round(dm, 3), "p": round(p, 4), "sig": p < 0.05}


# ─────────────────────────────────────────────
# БІР ГОРИЗОНТ — БАРЛЫҚ МОДЕЛЬДЕР
# NaN шешімі: SimpleImputer median стратегиясымен
# ─────────────────────────────────────────────
def run_horizon(feat_df, target_col):
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_")]

    # NaN бар жолдарды алып тастау (target бойынша)
    sub = feat_df.dropna(subset=[target_col]).copy()

    X = sub[feature_cols]
    y = sub[target_col].values

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y[:split], y[split:]

    # Imputer — NaN мәндерін медианмен толтыру
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    # Naive baseline: pm_lag_1 белгісін қолдану
    lag1_candidates = [c for c in feature_cols if "_lag_1" in c and
                       not any(k in c for k in ["temp", "wind", "humidity",
                                                  "blh", "no2", "aod", "dust",
                                                  "ms_", "precip"])]
    if lag1_candidates:
        lag1_idx = feature_cols.index(lag1_candidates[0])
        y_naive  = imputer.transform(X_te)[:, lag1_idx]
    else:
        y_naive  = np.full(len(y_te), np.mean(y_tr))

    results = {
        "Naive baseline": {
            "y_pred": y_naive, "y_test": y_te,
            **calc_metrics(y_te, y_naive)
        }
    }

    models = {
        "Ridge":             Ridge(alpha=1.0),
        "Random Forest":     RandomForestRegressor(
                                n_estimators=100, max_depth=15,
                                n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
                                n_estimators=100, max_depth=5,
                                learning_rate=0.1, subsample=0.8,
                                random_state=42),
    }

    fitted_models = {}
    for name, model in models.items():
        model.fit(X_tr_imp, y_tr)
        y_pred = model.predict(X_te_imp)
        results[name] = {
            "model": model,
            "y_pred": y_pred, "y_test": y_te,
            "feature_cols": feature_cols,
            "imputer": imputer,
            **calc_metrics(y_te, y_pred)
        }
        fitted_models[name] = model

    # DM test: GB vs RF
    e_gb = y_te - results["Gradient Boosting"]["y_pred"]
    e_rf = y_te - results["Random Forest"]["y_pred"]
    results["dm_test"] = dm_test(e_gb, e_rf)

    return results


# ─────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────
def run_shap(model, imputer, X_test_raw, feature_cols, pollutant, horizon):
    if not SHAP_OK:
        return
    print(f"   SHAP есептелуде ({pollutant}, {horizon})...")
    X_imp = imputer.transform(X_test_raw)
    X_df  = pd.DataFrame(X_imp, columns=feature_cols)
    sample = X_df.sample(min(500, len(X_df)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"SHAP — {pollutant.upper()} [{horizon}]",
                 fontsize=13, fontweight="bold")

    plt.sca(axes[0])
    shap.summary_plot(shap_values, sample, plot_type="bar",
                      max_display=15, show=False)
    axes[0].set_title("Feature importance (mean |SHAP|)")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, sample, max_display=15, show=False)
    axes[1].set_title("SHAP values (color = feature value)")

    plt.tight_layout()
    path = f"reports/figures/shap_{pollutant}_{horizon}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {path}")


# ─────────────────────────────────────────────
# ABLATION STUDY
# ─────────────────────────────────────────────
def run_ablation(feat_df, pollutant):
    feature_cols = [c for c in feat_df.columns if not c.startswith("target_")]
    sub  = feat_df.dropna(subset=["target_1h"]).copy()
    X_all = sub[feature_cols]
    y     = sub["target_1h"].values
    split = int(len(X_all) * 0.8)
    y_te  = y[split:]

    imputer = SimpleImputer(strategy="median")

    def test_cols(cols):
        avail = [c for c in cols if c in feature_cols]
        if not avail:
            return np.nan
        X_tr = imputer.fit_transform(X_all[avail].iloc[:split])
        X_te = imputer.transform(X_all[avail].iloc[split:])
        m = GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                       learning_rate=0.1, random_state=42)
        m.fit(X_tr, y[:split])
        return np.sqrt(mean_squared_error(y_te, m.predict(X_te)))

    lag_cols  = [c for c in feature_cols if f"{pollutant}_lag_" in c]
    roll_cols = [c for c in feature_cols if f"{pollutant}_roll_" in c or
                                             f"{pollutant}_diff_" in c]
    time_cols = [c for c in feature_cols if any(k in c for k in
                 ["hour_", "month_", "dow_", "day_of", "heating", "is_winter"])]
    era5_cols = [c for c in feature_cols if any(k in c for k in
                 ["temp", "wind", "humidity", "blh", "inversion", "calm",
                  "precip", "no2", "aod", "dust", "temp_x_wind", "pressure",
                  "cloud"])
                 and "ms_" not in c and "bias" not in c]
    ms_cols   = [c for c in feature_cols if "ms_" in c or "bias" in c]

    groups = [
        ("Тек lag_1",          lag_cols[:1]),
        ("+Барлық лагтар",     lag_cols),
        ("+Жылжымалы стат.",   lag_cols + roll_cols),
        ("+Циклдік уақыт",     lag_cols + roll_cols + time_cols),
        ("+ERA5 метео",        lag_cols + roll_cols + time_cols + era5_cols),
        ("+Meteostat станция", lag_cols + roll_cols + time_cols + era5_cols + ms_cols),
    ]

    rows = []
    for label, cols in groups:
        avail = [c for c in cols if c in feature_cols]
        if not avail:
            print(f"   {label:35s} → белгілер жоқ, өткізілді")
            continue
        rmse = test_cols(avail)
        rows.append({"Конфигурация": label, "n_белгі": len(avail), "RMSE": round(rmse, 3)})
        print(f"   {label:35s} → RMSE={rmse:.2f}  (n={len(avail)})")

    df_ab = pd.DataFrame(rows)
    has_ms = any("Meteostat" in r for r in df_ab["Конфигурация"])

    colors = []
    for lbl in df_ab["Конфигурация"]:
        if "ERA5" in lbl:
            colors.append("#534AB7")
        elif "Meteostat" in lbl:
            colors.append("#1D9E75")
        else:
            colors.append("#9FE1CB")

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(df_ab["Конфигурация"], df_ab["RMSE"], color=colors, height=0.55)
    for bar, val in zip(bars, df_ab["RMSE"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10)
    ax.set_xlabel("RMSE (мкг/м³)", fontsize=11)
    ax.set_title(f"Ablation Study — {pollutant.upper()} [+1h]", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, df_ab["RMSE"].max() * 1.18)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#9FE1CB", label="v1: PM лагтар + уақыт"),
        Patch(color="#534AB7", label="v2: +ERA5 метео"),
    ]
    if has_ms:
        legend_items.append(Patch(color="#1D9E75", label="v2.2: +Meteostat нақты станция"))
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = f"reports/figures/ablation_{pollutant}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {path}")
    df_ab.to_csv(f"reports/ablation_{pollutant}.csv", index=False)
    return df_ab


# ─────────────────────────────────────────────
# MULTI-HORIZON ГРАФИК
# ─────────────────────────────────────────────
def plot_multi_horizon(all_results, pollutant):
    horizons = [h for h in ["1h", "6h", "12h", "24h"] if h in all_results]
    labels   = [f"+{h}" for h in horizons]

    gb_rmse = [all_results[h]["Gradient Boosting"]["rmse"] for h in horizons]
    gb_r2   = [all_results[h]["Gradient Boosting"]["r2"]   for h in horizons]
    nb_rmse = [all_results[h]["Naive baseline"]["rmse"]    for h in horizons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Multi-Horizon болжам — {pollutant.upper()}, Алматы",
                 fontsize=13, fontweight="bold")

    x, w = np.arange(len(horizons)), 0.35
    axes[0].bar(x - w/2, gb_rmse, w, label="Gradient Boosting", color="#534AB7")
    axes[0].bar(x + w/2, nb_rmse, w, label="Naive baseline",    color="#B4B2A9")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("RMSE (мкг/м³)"); axes[0].set_title("RMSE")
    axes[0].legend(); axes[0].spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(gb_rmse):
        axes[0].text(i - w/2, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

    axes[1].bar(x, gb_r2, color="#1D9E75")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("R²"); axes[1].set_title("R² Score")
    axes[1].set_ylim(0, 1.05); axes[1].spines[["top", "right"]].set_visible(False)
    for i, v in enumerate(gb_r2):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = f"reports/figures/multi_horizon_{pollutant}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ {path}")


# ─────────────────────────────────────────────
# НЕГІЗГІ ФУНКЦИЯ
# ─────────────────────────────────────────────
def main():
    HORIZONS = {"1h": "target_1h", "6h": "target_6h",
                "12h": "target_12h", "24h": "target_24h"}

    all_metrics = []

    for pollutant, feat_file in [("pm25", "data/features_pm25_v2.csv"),
                                  ("pm10", "data/features_pm10_v2.csv")]:

        if not os.path.exists(feat_file):
            print(f"⚠  {feat_file} жоқ. Алдымен data_v2_openmeteo.py іске асырыңыз.")
            continue

        print(f"\n{'='*55}")
        print(f"  {pollutant.upper()}")
        print(f"{'='*55}")

        feat_df = pd.read_csv(feat_file, index_col=0, parse_dates=True)
        print(f"  Жиын: {len(feat_df)} жол, {len(feat_df.columns)} баған")

        # NaN статистикасы
        nan_pct = feat_df.isnull().mean() * 100
        high_nan = nan_pct[nan_pct > 50]
        if not high_nan.empty:
            print(f"  ⚠  50%+ NaN бар бағандар ({len(high_nan)}): {list(high_nan.index[:5])}")
            print(f"     → SimpleImputer(median) қолданылады")

        # Multi-horizon
        all_results = {}
        for h_label, t_col in HORIZONS.items():
            if t_col not in feat_df.columns:
                continue
            sub = feat_df.dropna(subset=[t_col])
            print(f"\n  Горизонт {h_label} ({len(sub)} бақылау)...")
            res = run_horizon(sub, t_col)
            all_results[h_label] = res

            dm = res["dm_test"]
            print(f"    GB   RMSE={res['Gradient Boosting']['rmse']:.2f}  "
                  f"MAE={res['Gradient Boosting']['mae']:.2f}  "
                  f"R²={res['Gradient Boosting']['r2']:.3f}  "
                  f"MAPE={res['Gradient Boosting']['mape']:.1f}%")
            print(f"    RF   RMSE={res['Random Forest']['rmse']:.2f}  "
                  f"R²={res['Random Forest']['r2']:.3f}")
            print(f"    DM-test p={dm['p']}  "
                  f"{'✓ маңызды (GB>RF)' if dm['sig'] and dm['stat'] > 0 else '○ маңызды емес'}")

            for mname in ["Naive baseline", "Ridge", "Random Forest", "Gradient Boosting"]:
                if mname not in res:
                    continue
                r = res[mname]
                all_metrics.append({
                    "pollutant": pollutant, "horizon": h_label,
                    "model": mname,
                    "rmse": r["rmse"], "mae": r["mae"],
                    "r2": r["r2"],     "mape": r["mape"]
                })

        # SHAP (1h)
        if "1h" in all_results and SHAP_OK:
            gb      = all_results["1h"]["Gradient Boosting"]
            imputer = gb["imputer"]
            fcols   = gb["feature_cols"]
            sub     = feat_df.dropna(subset=["target_1h"])
            split   = int(len(sub) * 0.8)
            X_te_raw = sub[fcols].iloc[split:]
            run_shap(gb["model"], imputer, X_te_raw, fcols, pollutant, "1h")

        # Ablation
        print(f"\n  Ablation study ({pollutant})...")
        run_ablation(feat_df, pollutant)

        # Multi-horizon график
        plot_multi_horizon(all_results, pollutant)

    # Барлық метрикаларды сақтау
    if all_metrics:
        df_m = pd.DataFrame(all_metrics)
        df_m.to_csv("reports/metrics_v2.csv", index=False)
        print(f"\n✓ Метрикалар: reports/metrics_v2.csv")
        print("\nGradient Boosting нәтижелері:")
        print(df_m[df_m["model"] == "Gradient Boosting"].to_string(index=False))

    print("\n✅ Modeling v2.0 аяқталды! reports/figures/ тексеріңіз.")


if __name__ == "__main__":
    main()
