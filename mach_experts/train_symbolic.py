import os
import numpy as np
from gplearn.genetic import SymbolicRegressor

import config

# ===== Ajustes =====
LATENT_FILE = os.path.join(config.RESULTS_DIR, "latents", "latents_z_train.npz")
OUT_EQ_DIR  = os.path.join(config.RESULTS_DIR, "equations_z")
os.makedirs(OUT_EQ_DIR, exist_ok=True)

MAX_SAMPLES = 300_000   # submuestreo para gplearn (ajusta si quieres)
RANDOM_SEED = 42

# gplearn settings (puedes endurecer parsimony si salen expresiones raras)
POP = 4000
GEN = 300
TOURN = 800
# ===================


def fit_one(dim: int, X, y, feature_names):
    reg = SymbolicRegressor(
        population_size=POP,
        generations=GEN,
        tournament_size=TOURN,
        function_set=("add", "sub", "mul", "div", "log", "sqrt"),
        metric="mse",
        parsimony_coefficient="auto",
        max_samples=0.9,
        p_crossover=0.7,
        p_subtree_mutation=0.02,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        feature_names=feature_names,
    )
    reg.fit(X, y)
    return reg


def main():
    if not os.path.exists(LATENT_FILE):
        raise FileNotFoundError(f"No existe {LATENT_FILE}. Ejecuta antes dump_latents_z.py")

    data = np.load(LATENT_FILE)
    X_raw = data["X_raw"].astype(np.float32)  # [N,9]
    Z = data["Z"].astype(np.float32)          # [N,3]

    N = X_raw.shape[0]
    print(f"[SYM] Cargado N={N} de {LATENT_FILE}")

    # Submuestreo
    n_sub = min(MAX_SAMPLES, N)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N, size=n_sub, replace=False)

    X = X_raw[idx]           # [n_sub,9]
    Zs = Z[idx]              # [n_sub,3]

    feature_names = ["x", "y", "z", "nx", "ny", "nz", "Mach", "AoA", "Pi"]

    # Entrenar z1,z2,z3
    for d in range(3):
        print(f"\n[SYM] Entrenando ecuación para z{d+1} ...")
        reg = fit_one(d, X, Zs[:, d], feature_names)

        eq = str(reg._program)
        out_txt = os.path.join(OUT_EQ_DIR, f"eq_z{d+1}.txt")
        with open(out_txt, "w") as f:
            f.write(eq)

        print(f"[SYM] z{d+1} = {eq}")
        print(f"[SAVED] {out_txt}")


if __name__ == "__main__":
    main()