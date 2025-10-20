import shap
import matplotlib.pyplot as plt

def compute_shap_values(model, X_sample, sample_limit=3000):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_sample[:sample_limit])

def plot_shap_summary(shap_values, feature_names, save_path):
    plt.figure(figsize=(8, 6))
    plt.rcParams['axes.unicode_minus'] = False
    shap.summary_plot(shap_values, shap_values, feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(8, 6)
    plt.subplots_adjust(left=0.15)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=18)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    colorbar = plt.gcf().axes[-1]
    colorbar.tick_params(labelsize=18)
    colorbar.set_ylabel(colorbar.get_ylabel(), fontsize=18)
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"[✔] SHAP summary saved to {save_path}")
