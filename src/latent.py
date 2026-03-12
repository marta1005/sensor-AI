import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

def latent_variable_ranking(all_z, X_raw, Y_raw=None, grad_cp=None, max_points=300000, seed=0):
    rng = np.random.default_rng(seed)
    N = all_z.shape[0]
    idx = rng.choice(N, size=min(max_points, N), replace=False)

    Z = all_z[idx]
    feats = {
        "x": X_raw[idx, 0], "y": X_raw[idx, 1], "z": X_raw[idx, 2],
        "nx": X_raw[idx, 3], "ny": X_raw[idx, 4], "nz": X_raw[idx, 5],
        "Mach": X_raw[idx, 6], "AoA": X_raw[idx, 7], "Pi": X_raw[idx, 8],
    }

    if Y_raw is not None:
        feats["Cp"]  = Y_raw[idx, 0]
        feats["Cfx"] = Y_raw[idx, 1]
        feats["Cfy"] = Y_raw[idx, 2]
        feats["Cfz"] = Y_raw[idx, 3]
        feats["|Cf|"] = np.linalg.norm(Y_raw[idx, 1:4], axis=1)

    if grad_cp is not None:
        feats["|∇Cp|"] = grad_cp[idx]
        feats["log|∇Cp|"] = np.log1p(grad_cp[idx])

    lr = LinearRegression()
    results = []

    for name, y in feats.items():
        y = y.reshape(-1)

        # R2 lineal
        lr.fit(Z, y)
        r2 = r2_score(y, lr.predict(Z))

        # Mutual Information (suma)
        mi = float(np.sum(mutual_info_regression(Z, y, random_state=0)))

        # Spearman: correlación por cada dimensión latente → tomamos el máximo absoluto
        if Z.shape[1] > 1:
            spears = [abs(spearmanr(Z[:, col], y)[0]) for col in range(Z.shape[1])]
            spear_max = max(spears) if spears else 0.0
        else:
            spear_max = abs(spearmanr(Z.ravel(), y)[0]) if Z.size > 0 else 0.0

        results.append((name, float(r2), mi, float(spear_max)))

    results.sort(key=lambda t: t[1], reverse=True)

    logger.info("\n[LATENT DIAG] Top variables explicadas por Z (orden por R²):")
    for name, r2, mi, spear in results[:12]:
        logger.info(f"  {name:12s} | R2={r2: .4f} | MI_sum={mi: .4f} | max|Spearman|={spear:.4f}")

    return results