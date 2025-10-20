import numpy as np
import rasterio
from windows import multi_channel_dominant_filter_Intersection, clean_X_y, multi_channel_dominant_filter
from pathlib import Path

np.random.seed(42)

def prepare_dataset(region="wencheng"):
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    if region.lower() == "wenchang":
        label_map = np.load("E:/fuwuqibeifen2/save_code_250615/label_map_wc_hthc_250615.npy")
        hc_map = np.load("E:/fuwuqibeifen2/save_code_250615/HC_map_wc_hthc_250615.npy")
        ht_map = np.load("E:/fuwuqibeifen2/save_code_250615/HT_map_wc_hthc_250615.npy")
        input_map = np.load("E:/fuwuqibeifen2/save_code_250615/input_map_wc_hthc_250615.npy") / 255.0
        biomass_path = "H:/SAR_Lidar/hainan_data/biomass_roi_resize.dat"
        channel_indices = [7, 9, 10]
        bins, top_k = 30, 5
        noise_std = 1.75
    else:
        label_map = np.load("E:/fuwuqibeifen2/save_for_biomass/label_map_gzl.npy")
        hc_map = np.load("E:/fuwuqibeifen2/save_for_biomass/HC_map_gzl.npy")
        ht_map = np.load("E:/fuwuqibeifen2/save_for_biomass/HT_map_gzl.npy")
        input_map = np.load("E:/fuwuqibeifen2/save_for_biomass/input_map_gzl.npy") / 255.0
        biomass_path = "H:/SAR_Lidar/gongzhuling_cm2_202405/gongzhuling_s2_roi_resize.dat"
        channel_indices = [8, 9, 11]
        bins, top_k = 100, 5
        noise_std = 0.5

    with rasterio.open(biomass_path) as src:
        biomass_map = src.read(1)

    input_map = input_map.reshape(-1, 9)
    label_map = label_map.reshape(-1, 1)
    hc_map = hc_map.reshape(-1, 1)
    ht_map = ht_map.reshape(-1, 1)
    X = np.concatenate([input_map, label_map, hc_map, ht_map], axis=1)
    y = biomass_map.reshape(-1)

    X_clean, y_clean = clean_X_y(X, y)

    X_filtered, y_filtered = multi_channel_dominant_filter_Intersection(
        X_clean, y_clean, channel_indices=channel_indices, bins=bins, top_k=top_k
    )

    X_prior, y_prior = multi_channel_dominant_filter(
        X_clean, y_clean, channel_indices=[9], bins=100, top_k=7#8
    )

    n1 = int(len(X_filtered) * 0.9)#1
    n2 = int(len(X_prior) * 0.1)#0.15
    idx1 = np.random.choice(len(X_filtered), n1, replace=False)
    idx2 = np.random.choice(len(X_prior), n2, replace=False)

    X_final = np.vstack([X_filtered[idx1], X_prior[idx2]])
    y_final = np.hstack([y_filtered[idx1], y_prior[idx2]])
    y_final = y_final + np.random.normal(0, noise_std, size=len(y_final))

    np.save(data_dir / f"{region}_X.npy", X_final)
    np.save(data_dir / f"{region}_y.npy", y_final)
    print(f"[✔] Dataset saved: {X_final.shape[0]} samples, {X_final.shape[1]} features")

if __name__ == "__main__":
    prepare_dataset(region="wenchang")#wenchang/gzl
    print('done')
