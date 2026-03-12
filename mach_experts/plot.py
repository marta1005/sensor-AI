import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

# ================== CONFIGURACIÓN ==================
DATA_DIR = "../data_cut_y12"   # carpeta con X_cut_train.npy, Y_cut_train.npy, idx_keep_train.npy
SNAPSHOT_IDX = 0              # snapshot que quieres ver

# Cómo definimos el ala derecha:
#   - WING_AXIS: 'z' si el span está en Z, 'y' si el span está en Y
#   - WING_SIGN: 'positive' -> coord > 0, 'negative' -> coord < 0
WING_AXIS = "z"
WING_SIGN = "positive"

FOLDER_IMAGE = None           # None -> mostrar en pantalla, "./figs" -> guardar PNGs
# ===================================================


def createFigure_TB(X, Y, Z, pres, aeroCond, titlePrefix,
                    folderImage=None, error_flag=False):
    """
    Versión simplificada:
      - SOLO pinta Cp (una métrica).
      - Usa scatter3D (dos vistas).
      - Una sola barra de color.

    Parámetros:
      X, Y, Z   : arrays 1D de coords (solo el ala que quieras mostrar).
      pres      : Cp en una de estas formas:
                    (N,), (N,1) o (N,4) -> usa pres[:,0]
      aeroCond  : (Mach, AoA, Pi)
      titlePref.: string para el título ("CFD", etc.)
    """

    # --- Condiciones aerodinámicas ---
    sMinf, sAoA, pressure = aeroCond

    # --- Cp: aceptar diferentes formas de 'pres' ---
    pres = np.asarray(pres)
    if pres.ndim == 1:
        cp_vals = pres
    elif pres.ndim == 2:
        if pres.shape[1] == 1:
            cp_vals = pres[:, 0]
        else:
            # Si viene [Cp, Cfx, Cfy, Cfz], nos quedamos solo con Cp
            cp_vals = pres[:, 0]
    else:
        raise ValueError(f"'pres' debe tener shape (N,), (N,1) o (N,4). Recibido: {pres.shape}")

    # --- Ajustes de cámara y estilos ---
    cameraSettings = {
        'elevation': 90.,
        'azimuth': 0.,
        'zoom': 1.9,
        'xlim': [None, None],
        'ylim': [None, None],
        'zlim': [None, None],
        'xoffsets': [0., 0.],
        'yoffsets': [0., 4.],
        'zoffsets': [0., 0.]
    }

    plotSettings = {
        'figsize': (12, 4),
        'dpi': 400,
        'title': '{:s} $p_i$ = {:.1f} 10⁵, $M_{{\\infty}}$ = {:.2f}, AoA = {:.1f}°'
                 .format(titlePrefix, pressure, sMinf, sAoA),
        'title_fontsize': 12,
        'cmap': 'jet',
        'cbar': {
            'title_fontsize': 8,
            'ticks_fontsize': 6,
        }
    }

    xmin, xmax = np.min(X), np.max(X)
    ymin, ymax = np.min(Y), np.max(Y)
    zmin, zmax = np.min(Z), np.max(Z)

    # Rango de colores (Cp o error de Cp)
    if error_flag:
        vmin, vmax = 0.0, 1.0
        cbar_label = 'relative error $C_p$'
    else:
        vmin, vmax = -1.0, 1.0
        cbar_label = '$C_p$'

    if cameraSettings['xlim'][0] is None: cameraSettings['xlim'][0] = xmin
    if cameraSettings['xlim'][1] is None: cameraSettings['xlim'][1] = xmax
    if cameraSettings['ylim'][0] is None: cameraSettings['ylim'][0] = ymin
    if cameraSettings['ylim'][1] is None: cameraSettings['ylim'][1] = ymax
    if cameraSettings['zlim'][0] is None: cameraSettings['zlim'][0] = zmin
    if cameraSettings['zlim'][1] is None: cameraSettings['zlim'][1] = zmax

    # ========= FIGURA =========
    if plotSettings['figsize'] is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=plotSettings['figsize'])

    # 2 filas (scatter + colorbar), 2 columnas (dos vistas)
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        height_ratios=[20, 1],
    )

    scatters = []
    for j in range(2):
        ax = fig.add_subplot(gs[0, j], projection=Axes3D.name)

        sca = ax.scatter3D(
            X, Y, Z,
            vmin=vmin, vmax=vmax,
            c=cp_vals,
            cmap=plt.get_cmap(plotSettings['cmap']),
            clip_on=False,
            s=2.0,
            alpha=0.9
        )
        scatters.append(sca)

        elev = cameraSettings['elevation'] * ((-1) ** (j + 1))
        azim = cameraSettings['azimuth'] + 180 * np.abs(j - 1)
        ax.view_init(elev=elev, azim=azim)

        xlim = (cameraSettings['xlim'][0] + cameraSettings['xoffsets'][0],
                cameraSettings['xlim'][1] + cameraSettings['xoffsets'][1])
        ylim = (cameraSettings['ylim'][0] + cameraSettings['yoffsets'][0],
                cameraSettings['ylim'][1] + cameraSettings['yoffsets'][1])
        zlim = (cameraSettings['zlim'][0] + cameraSettings['zoffsets'][0],
                cameraSettings['zlim'][1] + cameraSettings['zoffsets'][1])
        ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
        ax.set_axis_off()

        limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
        ax.set_box_aspect(np.ptp(limits, axis=1), zoom=cameraSettings['zoom'])

        if j == 0:
            ax.text((xlim[1] - xlim[0]) / 4 + 0.5,
                    (ylim[1] - ylim[0]) / 2 + 7, 0,
                    'bottom', fontsize=6)
        else:
            ax.text((xlim[1] - xlim[0]) / 4,
                    (ylim[1] - ylim[0]) / 2 - 5, 0,
                    'top', fontsize=6)

    # Colorbar única
    cax = fig.add_subplot(gs[1, :])
    cbar = fig.colorbar(scatters[0], cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label,
                   size=plotSettings['cbar']['title_fontsize'])
    cbar.ax.tick_params(labelsize=plotSettings['cbar']['ticks_fontsize'])
    cbar.ax.xaxis.set_label_position('top')

    # Título
    title = plotSettings.get('title', None)
    if title is not None:
        fig.suptitle(plotSettings['title'],
                     fontsize=plotSettings['title_fontsize'])
    fig.tight_layout(h_pad=2.0)

    if folderImage:
        os.makedirs(folderImage, exist_ok=True)
        fileName = (folderImage +
                    '/{:s}fig_Pi{:.1f}Minf{:.2f}AoA{:.1f}.png'
                    .format(titlePrefix, pressure, sMinf, sAoA))
        fig.savefig(fileName, dpi=plotSettings['dpi'])
        plt.close()
        print(f"[SAVED] {fileName}")
    else:
        plt.show()


def main():
    # ---------- Cargar datos cortados ----------
    x_path   = os.path.join(DATA_DIR, "X_cut_train.npy")
    y_path   = os.path.join(DATA_DIR, "Y_cut_train.npy")
    idx_path = os.path.join(DATA_DIR, "idx_keep_train.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"No se encuentra {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"No se encuentra {y_path}")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"No se encuentra {idx_path}")

    X_cut    = np.load(x_path).astype(np.float32)   # [N_total_cut, n_feat]
    Y_cut    = np.load(y_path).astype(np.float32)   # [N_total_cut, 4]
    idx_keep = np.load(idx_path).astype(np.int64)   # [NP_cut]

    N_total_cut, n_feat = X_cut.shape
    N_total_y, n_out    = Y_cut.shape
    NP_cut = idx_keep.shape[0]

    assert N_total_cut == N_total_y, "X_cut y Y_cut deben tener el mismo N"
    if N_total_cut % NP_cut != 0:
        raise ValueError(
            f"No cuadra N_total_cut={N_total_cut} con NP_cut={NP_cut} "
            f"(N_total_cut % NP_cut = {N_total_cut % NP_cut})"
        )

    nwallp = NP_cut
    n_snapshots = N_total_cut // nwallp

    print(f"[INFO] N_total_cut = {N_total_cut}")
    print(f"[INFO] NP_cut (nwallp) = {nwallp}")
    print(f"[INFO] n_snapshots = {n_snapshots}")

    if SNAPSHOT_IDX < 0 or SNAPSHOT_IDX >= n_snapshots:
        raise ValueError(f"SNAPSHOT_IDX={SNAPSHOT_IDX} fuera de rango [0, {n_snapshots-1}]")

    # ---------- Extraer snapshot ----------
    X_cond = X_cut[SNAPSHOT_IDX * nwallp:(SNAPSHOT_IDX + 1) * nwallp, :]
    Y_cond = Y_cut[SNAPSHOT_IDX * nwallp:(SNAPSHOT_IDX + 1) * nwallp, :]  # [N,4]

    print(f"[INFO] Snapshot {SNAPSHOT_IDX}: X_cond shape={X_cond.shape}, Y_cond shape={Y_cond.shape}")

    # aeroCond = (Mach, AoA, Pi) → últimas 3 columnas de X_cond
    aeroCond = X_cond[0, -3:]
    print("aero condition (Mach, AoA, Pi) =", aeroCond)

    # Coordenadas
    X = X_cond[:, 0]
    Y = X_cond[:, 1]
    Z = X_cond[:, 2]

    # ---------- Filtro ala derecha ----------
    if WING_AXIS.lower() == "z":
        span_coord = Z
    elif WING_AXIS.lower() == "y":
        span_coord = Y
    else:
        raise ValueError("WING_AXIS debe ser 'y' o 'z'")

    if WING_SIGN == "positive":
        mask_right = span_coord > 0.0
    elif WING_SIGN == "negative":
        mask_right = span_coord < 0.0
    else:
        raise ValueError("WING_SIGN debe ser 'positive' o 'negative'")

    X_r = X[mask_right]
    Y_r = Y[mask_right]
    Z_r = Z[mask_right]
    Y_cond_r = Y_cond[mask_right, :]   # [N_right, 4]

    print(f"[INFO] Ala derecha: N = {len(X_r)} puntos")

    # ---------- SOLO Cp ----------
    Cp_r = Y_cond_r[:, 0]   # vector (N_right,)

    # ---------- Figura ----------
    createFigure_TB(
        X_r,
        Y_r,
        Z_r,
        Cp_r,                 # sólo Cp
        aeroCond,
        titlePrefix="CFD ala derecha (Cp)",
        folderImage=FOLDER_IMAGE,
        error_flag=False
    )


if __name__ == "__main__":
    main()