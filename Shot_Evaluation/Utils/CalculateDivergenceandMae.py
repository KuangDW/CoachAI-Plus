import numpy as np
from scipy.special import rel_entr
from scipy.spatial.distance import jensenshannon

def calculate_divergences_and_mae(dict1, dict2):
    """Calculate KL divergence, JS divergence, and MAE between two dictionaries."""
    total1 = sum(dict1.values())
    total2 = sum(dict2.values())

    p = np.array([dict1.get(k, 0) / total1 for k in sorted(set(dict1) | set(dict2))])
    q = np.array([dict2.get(k, 0) / total2 for k in sorted(set(dict1) | set(dict2))])

    # Avoid division by zero for KL and JS calculation
    q = np.where(q == 0, np.finfo(float).eps, q)
    p = np.where(p == 0, np.finfo(float).eps, p)

    kl_div = sum(rel_entr(p, q)) # KL divergence
    js_div = jensenshannon(p, q) ** 2 # JS divergence
    mae = np.mean(np.abs(p - q)) # MAE

    print(f'KL: {round(kl_div, 4)}, JS: {round(js_div, 4)}, MAE: {round(mae, 4)}')

    return kl_div, js_div, mae
