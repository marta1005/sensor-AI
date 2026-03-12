import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from . import config
import logging

logger = logging.getLogger(__name__)

def plot_latent_space_and_save(ae_model, data_tensor, gmm_labels, grad_values):
    """
    Visualiza el espacio latente:
      - 3D clusters vs GMM
      - 3D coloreado por gradiente
      - Proyecciones 2D (Z1-Z2, Z1-Z3, Z2-Z3) por clusters y gradiente
      - Histogramas 1D de Z1, Z2, Z3 por cluster

    Soporta latent_dim >= 2. Si latent_dim > 3, usa las 3 primeras componentes.
    """
    logger.info("[VIS] Generando gráficos de Espacio Latente...")

    data_tensor = np.asarray(data_tensor)
    gmm_labels = np.asarray(gmm_labels).reshape(-1)
    grad_values = np.asarray(grad_values).reshape(-1)

    # --- Submuestreo para no morir con millones de puntos ---
    num_samples = 400000
    n_total = len(data_tensor)
    if n_total > num_samples:
        indices = np.random.choice(n_total, num_samples, replace=False)
    else:
        indices = np.arange(n_total)

    sample_tensor = torch.from_numpy(data_tensor[indices]).to(config.DEVICE, non_blocking=True)
    sample_labels = gmm_labels[indices]
    sample_grads = grad_values[indices]

    # --- Obtener z ---
    ae_model.eval()
    with torch.no_grad():
        _, z = ae_model(sample_tensor)
        z_np = z.detach().cpu().numpy()

    latent_dim = z_np.shape[1]
    if latent_dim < 2:
        raise ValueError(f"El espacio latente tiene dim={latent_dim}. Necesitas al menos 2D para visualizar.")

    # Si latent_dim > 3 usamos solo las 3 primeras
    if latent_dim > 3:
        z_np = z_np[:, :3]
        latent_dim = 3

    # Nombres de ejes
    axis_names = [f"Z{i+1}" for i in range(latent_dim)]

    # --------------------------
    # FIGURA 1: 3D clusters & grad
    # --------------------------
    if latent_dim == 2:
        # Solo 2D, mantenemos compatibilidad
        fig = plt.figure(figsize=(18, 8))

        ax1 = fig.add_subplot(121)
        sc1 = ax1.scatter(z_np[:, 0], z_np[:, 1],
                          c=sample_labels, cmap='coolwarm', s=2, alpha=0.6)
        ax1.set_title("Latente (2D) - Clusters GMM")
        ax1.set_xlabel(axis_names[0])
        ax1.set_ylabel(axis_names[1])

        ax2 = fig.add_subplot(122)
        sc2 = ax2.scatter(z_np[:, 0], z_np[:, 1],
                          c=np.log1p(sample_grads), cmap='viridis', s=2, alpha=0.6)
        ax2.set_title("Latente (2D) - log(1+|∇Cp|)")
        ax2.set_xlabel(axis_names[0])
        ax2.set_ylabel(axis_names[1])
        cbar = plt.colorbar(sc2, ax=ax2, shrink=0.85)
        cbar.set_label('log(1+|∇Cp|)')

        plt.tight_layout()
        path = os.path.join(config.RESULTS_DIR, "figures", "latent_space_2d.png")
        plt.savefig(path, dpi=300)
        logger.info(f"[VIS] Figura 2D guardada en {path}")
        plt.close()

    else:
        # 3D
        fig = plt.figure(figsize=(18, 8))

        # PANEL 1: Clusters GMM
        ax1 = fig.add_subplot(121, projection='3d')
        sc1 = ax1.scatter(z_np[:, 0], z_np[:, 1], z_np[:, 2],
                          c=sample_labels, cmap='coolwarm', s=2, alpha=0.6)
        ax1.set_title("Espacio Latente 3D: Clusters GMM")
        ax1.set_xlabel(axis_names[0])
        ax1.set_ylabel(axis_names[1])
        ax1.set_zlabel(axis_names[2])
        ax1.view_init(elev=25, azim=-130)

        # PANEL 2: Gradiente
        ax2 = fig.add_subplot(122, projection='3d')
        sc2 = ax2.scatter(z_np[:, 0], z_np[:, 1], z_np[:, 2],
                          c=np.log1p(sample_grads), cmap='viridis', s=2, alpha=0.6)
        ax2.set_title("Espacio Latente 3D: log(1+|∇Cp|)")
        ax2.set_xlabel(axis_names[0])
        ax2.set_ylabel(axis_names[1])
        ax2.set_zlabel(axis_names[2])
        ax2.view_init(elev=25, azim=-130)
        cbar = plt.colorbar(sc2, ax=ax2, shrink=0.85)
        cbar.set_label('log(1+|∇Cp|)')

        plt.tight_layout()
        path3d = os.path.join(config.RESULTS_DIR, "figures", "latent_space_3d.png")
        plt.savefig(path3d, dpi=300)
        logger.info(f"[VIS] Figura 3D guardada en {path3d}")
        plt.close()

    # --------------------------
    # FIGURA 2: Proyecciones 2D (clusters & grad)
    # --------------------------
    if latent_dim >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        pairs = [(0, 1), (0, 2), (1, 2)]
        titles = [f"{axis_names[i]} vs {axis_names[j]}" for (i, j) in pairs]

        # Fila 1: clusters
        for ax, (i, j), title in zip(axes[0], pairs, titles):
            sc = ax.scatter(z_np[:, i], z_np[:, j],
                            c=sample_labels, cmap='coolwarm', s=2, alpha=0.6)
            ax.set_xlabel(axis_names[i])
            ax.set_ylabel(axis_names[j])
            ax.set_title(f"Clusters GMM: {title}")

        # Fila 2: gradiente
        for ax, (i, j), title in zip(axes[1], pairs, titles):
            sc = ax.scatter(z_np[:, i], z_np[:, j],
                            c=np.log1p(sample_grads), cmap='viridis', s=2, alpha=0.6)
            ax.set_xlabel(axis_names[i])
            ax.set_ylabel(axis_names[j])
            ax.set_title(f"log(1+|∇Cp|): {title}")
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('log(1+|∇Cp|)')

        plt.tight_layout()
        path2d = os.path.join(config.RESULTS_DIR, "figures", "latent_space_projections.png")
        plt.savefig(path2d, dpi=300)
        logger.info(f"[VIS] Proyecciones 2D guardadas en {path2d}")
        plt.close()

        # --------------------------
        # FIGURA 3: Histogramas 1D por cluster
        # --------------------------
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        clusters = np.unique(sample_labels)
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']

        for dim_idx, ax in enumerate(axes):
            for c_idx, c in enumerate(clusters):
                mask = (sample_labels == c)
                ax.hist(
                    z_np[mask, dim_idx],
                    bins=80,
                    alpha=0.5,
                    density=True,
                    color=colors[c_idx % len(colors)],
                    label=f"Cluster {int(c)}" if dim_idx == 0 else None,
                )
            ax.set_title(f"Distribución {axis_names[dim_idx]} por cluster")
            ax.set_xlabel(axis_names[dim_idx])
            ax.set_ylabel("Densidad")

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")

        plt.tight_layout(rect=[0, 0, 0.96, 1])
        path_hist = os.path.join(config.RESULTS_DIR, "figures", "latent_histograms.png")
        plt.savefig(path_hist, dpi=300)
        logger.info(f"[VIS] Histogramas de Z guardados en {path_hist}")
        plt.close()


def visualize_snapshot_and_save(model, data_handler, snapshot_idx=0):
    logger.info(f"[VIS] Visualizando snapshot índice {snapshot_idx}...")
    
    # Calcular índices para extraer UN solo avión
    start = snapshot_idx * config.NP
    end = start + config.NP
    
    # Extraer datos
    X_slice = data_handler.X_final[start:end]
    coords = data_handler.X_raw[start:end, 0:3]
    
    # Inferencia
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_slice).to(config.DEVICE)
        preds, sensor_prob = model(x_tensor)
        
        preds = preds.cpu().numpy()
        sensor_prob = sensor_prob.cpu().numpy()

    # Asumiendo que el código truncado era para plotear; completo con ejemplo básico
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=preds[:, 0], cmap='jet')
    ax1.set_title('Cp Predicho')
    plt.colorbar(sc1, ax=ax1)

    ax2 = fig.add_subplot(132, projection='3d')
    sc2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=sensor_prob.squeeze(), cmap='viridis')
    ax2.set_title('Probabilidad de Choque [0-1]')
    plt.colorbar(sc2, ax=ax2)

    if data_handler.has_targets:
        ax3 = fig.add_subplot(133, projection='3d')
        sc3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=data_handler.Y_raw[start:end, 0], cmap='jet')
        ax3.set_title('Cp Real')
        plt.colorbar(sc3, ax=ax3)

    plt.tight_layout()
    
    filename = f'aircraft_prediction_idx{snapshot_idx}.png'
    path = os.path.join(config.RESULTS_DIR, 'figures', filename)
    plt.savefig(path, dpi=300)
    logger.info(f"[VIS] Figura guardada en {path}")
    plt.close()

def visualize_cp_real_pred_error_and_save(
    model,
    data_handler,
    snapshot_idx=0,
    invert_cp=True,
    error_mode="signed"  # "signed" (pred-real) o "abs"
):
    logger.info(f"[VIS] Cp REAL vs PRED + ERROR para snapshot {snapshot_idx}...")

    NP = config.NP
    start = snapshot_idx * NP
    end = start + NP

    coords = data_handler.X_raw[start:end, 0:3]
    X_slice = data_handler.X_final[start:end]

    # --- Predicción ---
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_slice).to(config.DEVICE)
        y_pred_norm, _ = model(x_tensor)  # [NP,4]
        y_pred_norm = y_pred_norm.cpu().numpy()

    if data_handler.scaler_y is None:
        raise ValueError("Para visualizar Cp predicho en físico necesitas data_handler.scaler_y.")
    y_pred = data_handler.scaler_y.inverse_transform(y_pred_norm)
    cp_pred = y_pred[:, 0]

    # --- Cp real ---
    if not getattr(data_handler, "has_targets", False) or data_handler.Y_raw is None:
        raise ValueError("Para Cp REAL necesitas cargar y_path (Y_test.npy) en el data_handler.")
    cp_real = data_handler.Y_raw[start:end, 0]

    # Convención típica: -Cp
    if invert_cp:
        cp_real_plot = -cp_real
        cp_pred_plot = -cp_pred
        title_suffix = "(-Cp)"
    else:
        cp_real_plot = cp_real
        cp_pred_plot = cp_pred
        title_suffix = "(Cp)"

    # --- Error ---
    err = cp_pred_plot - cp_real_plot  # coherente con lo que se muestra
    if error_mode == "abs":
        err_plot = np.abs(err)
    else:
        err_plot = err

    # Misma escala para Cp real y pred
    vmin_cp = float(min(cp_real_plot.min(), cp_pred_plot.min()))
    vmax_cp = float(max(cp_real_plot.max(), cp_pred_plot.max()))

    # Escala del error
    if error_mode == "abs":
        vmin_err = 0.0
        vmax_err = float(err_plot.max())
    else:
        max_abs = float(np.max(np.abs(err_plot)))
        vmin_err = -max_abs
        vmax_err = max_abs

    fig = plt.figure(figsize=(22, 7))

    # --- Panel 1: Cp real ---
    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      c=cp_real_plot, s=1, vmin=vmin_cp, vmax=vmax_cp, cmap="jet")
    ax1.set_title(f"Cp REAL {title_suffix}")
    ax1.view_init(elev=30, azim=-120)
    plt.colorbar(sc1, ax=ax1, fraction=0.03)

    # --- Panel 2: Cp predicho ---
    ax2 = fig.add_subplot(132, projection="3d")
    sc2 = ax2.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      c=cp_pred_plot, s=1, vmin=vmin_cp, vmax=vmax_cp, cmap="jet")
    ax2.set_title(f"Cp PREDICHO {title_suffix}")
    ax2.view_init(elev=30, azim=-120)
    plt.colorbar(sc2, ax=ax2, fraction=0.03)

    # --- Panel 3: Error ---
    ax3 = fig.add_subplot(133, projection="3d")
    cmap_err = "magma" if error_mode == "abs" else "coolwarm"
    sc3 = ax3.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                      c=err_plot, s=1, vmin=vmin_err, vmax=vmax_err, cmap=cmap_err)
    if error_mode == "abs":
        ax3.set_title(f"ERROR ABS |Pred-Real| {title_suffix}")
    else:
        ax3.set_title(f"ERROR (Pred-Real) {title_suffix}")
    ax3.view_init(elev=30, azim=-120)
    plt.colorbar(sc3, ax=ax3, fraction=0.03)

    plt.tight_layout()

    out_path = os.path.join(config.RESULTS_DIR, "figures", f"cp_real_pred_error_idx{snapshot_idx}.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info(f"[VIS] Figura guardada en: {out_path}")
    return out_path