import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from plot_figure import plot_fig
from shap_utils import compute_shap_values, plot_shap_summary
from pathlib import Path

np.random.seed(42)


def train_and_evaluate(region, feature_indices_to_keep, feature_names, suffix):
    """Train, evaluate, and plot results for a specific feature combination."""
    data_dir = Path("data")
    output_dir = Path("outputs") / region
    output_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(data_dir / f"{region}_X.npy")
    y = np.load(data_dir / f"{region}_y.npy")

    # 特征筛选
    X = X[:, feature_indices_to_keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        min_samples_split=5, min_samples_leaf=3,
        max_features="sqrt", bootstrap=False,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    corr, _ = pearsonr(y_test, y_pred)

    print(f"\n===== {region.upper()} | {suffix} =====")
    print(f"Features: {feature_names}")
    print(f"R = {corr:.3f} | RMSE = {rmse:.3f} | MAE = {mae:.3f}")

    # 绘制预测散点图
    scatter_path = output_dir / f"prediction_scatter_{suffix}.png"
    plot_fig(
        y_all=y, true_value=y_test, pred_value=y_pred,
        RMSE=rmse, MAE=mae, correlation=corr,
        path=scatter_path, name=f"RF ({suffix})"
    )

    # SHAP 分析
    shap_values = compute_shap_values(model, X_test, sample_limit=3000)
    shap_path = output_dir / f"shap_summary_{suffix}.png"
    plot_shap_summary(shap_values, feature_names, save_path=shap_path)


def main(region="wenchang", mode="full"):
    """
    mode 可选：
        "full"              — 全部特征
        "no_H"              — 去掉 H
        "no_HcHt"           — 去掉 hc 和 ht
        "no_all_heights"    — 去掉 H、hc、ht
    """
    all_indices = list(range(12))

    feature_configs = {
        "full": (
            all_indices,
            ['R','G','B','C-HH','C-HV','C-VV','L-HH','L-HV','L-VV','H','$h_c$','$h_t$']
        ),
        "no_H": (
            [i for i in all_indices if i != 9],
            ['R','G','B','C-HH','C-HV','C-VV','L-HH','L-HV','L-VV','$h_c$','$h_t$']
        ),
        "no_HcHt": (
            [i for i in all_indices if i not in [10, 11]],
            ['R','G','B','C-HH','C-HV','C-VV','L-HH','L-HV','L-VV','H']
        ),
        "no_all_heights": (
            [i for i in all_indices if i not in [9, 10, 11]],
            ['R','G','B','C-HH','C-HV','C-VV','L-HH','L-HV','L-VV']
        ),
    }

    if mode not in feature_configs:
        raise ValueError(f"Invalid mode '{mode}'. Choose from {list(feature_configs.keys())}")

    indices, names = feature_configs[mode]
    train_and_evaluate(region, indices, names, mode)


if __name__ == "__main__":
    # 只需改这里的参数即可切换实验模式：
    # 可选 "wenchang", "gzl"
    # 可选 "full", "no_H", "no_HcHt", "no_all_heights"
    main(region="wenchang", mode="no_HcHt")
    print("✅ Experiment completed.")
