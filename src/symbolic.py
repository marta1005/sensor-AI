from gplearn.genetic import SymbolicClassifier
import numpy as np
import os
from . import config
import logging

logger = logging.getLogger(__name__)


def get_interpretable_sensor_module(X_data, shock_labels, var_names=None):
    """
    Entrena un clasificador simbólico con gplearn para detectar zonas de choque.

    MUY IMPORTANTE (tu requisito):
      - Input: SOLO geometría + condiciones de vuelo (X_data).
      - Target: shock_labels (0/1) definidos por la caja negra (AE + GMM + física).
      - En inferencia, este sensor solo necesita X_data (x,y,z,nx,ny,nz,Mach,AoA,Pi).

    No usa Cp ni gradientes ni z latente.
    """
    logger.info("\n[gplearn] Entrenando sensor simbólico (clasificación 0/1) a partir de X_físico...")

    X_data = np.asarray(X_data, dtype=np.float32)
    y = np.asarray(shock_labels).astype(int).ravel()

    if var_names is not None:
        logger.info(f"[gplearn] Variables de entrada: {var_names}")

    # Submuestreo aleatorio (para no saturar memoria/tiempo)
    n_sub = min(1_000_000, len(X_data))
    idx = np.random.choice(len(X_data), n_sub, replace=False)
    X_sub = X_data[idx]
    y_sub = y[idx]

    # Calcular pesos para compensar desbalance
    n_total = len(y_sub)
    n_shock = np.sum(y_sub == 1)
    n_no_shock = n_total - n_shock

    if n_shock == 0:
        raise ValueError("No hay ejemplos de choque en la submuestra. Revisa los datos o sube n_sub.")

    weight_shock = n_total / (2.0 * n_shock)
    weight_no_shock = n_total / (2.0 * n_no_shock)

    sample_weight = np.where(y_sub == 1, weight_shock, weight_no_shock).astype(np.float32)

    logger.info(f"[gplearn] Submuestra: N={n_sub} | shock%={100*n_shock/n_total:.2f}%")
    logger.info(f"                  | peso choque={weight_shock:.3f} | peso no-choque={weight_no_shock:.3f}")

    # Conjunto de funciones más "físico" (evitamos cosas demasiado marcianas)
    function_set = ("add", "sub", "mul", "div", "log")

    clf = SymbolicClassifier(
        population_size=5000,
        generations=600,
        tournament_size=1200,
        function_set=function_set,
        parsimony_coefficient="auto",
        max_samples=0.9,
        metric="log loss",
        p_crossover=0.7,
        p_subtree_mutation=0.01,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(X_sub, y_sub, sample_weight=sample_weight)

    eq_str = str(clf._program)
    os.makedirs(os.path.join(config.RESULTS_DIR, "equations"), exist_ok=True)
    out_path = os.path.join(config.RESULTS_DIR, "equations", "sensor_equation_gplearn.txt")
    with open(out_path, "w") as f:
        f.write(eq_str)

    logger.info(f"[gplearn] Programa CLASIFICADOR guardado en: {out_path}")
    logger.info(f"[gplearn] Ecuación encontrada (clasificador): {eq_str}")
    return clf
