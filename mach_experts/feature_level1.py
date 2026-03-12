import numpy as np

def add_level1_features(X_raw_9: np.ndarray) -> np.ndarray:
    """
    X_raw_9: [N,9] = [x,y,z,nx,ny,nz,Mach,AoA,Pi] en físico (sin normalizar)

    Devuelve X_aug_raw: [N,19] = X_raw_9 + 10 features nivel 1.
    """
    X = X_raw_9
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]
    nx = X[:, 3:4]
    ny = X[:, 4:5]
    nz = X[:, 5:6]
    mach = X[:, 6:7]
    aoa = X[:, 7:8]
    # pi = X[:, 8:9]  # no hace falta separarla

    # Features nuevas
    sin_aoa = np.sin(aoa)
    cos_aoa = np.cos(aoa)
    mach2 = mach ** 2
    r = np.sqrt(y ** 2 + z ** 2)

    abs_nx = np.abs(nx)
    abs_ny = np.abs(ny)
    abs_nz = np.abs(nz)

    mach_nx = mach * nx
    mach_ny = mach * ny
    mach_nz = mach * nz

    extra = np.concatenate(
        [sin_aoa, cos_aoa, mach2, r, abs_nx, abs_ny, abs_nz, mach_nx, mach_ny, mach_nz],
        axis=1
    ).astype(np.float32)

    X_aug = np.concatenate([X, extra], axis=1).astype(np.float32)  # [N,19]
    return X_aug