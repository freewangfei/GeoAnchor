from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    if mask is not None:
        valid = mask.astype(bool).ravel()
        yt = y_true.astype(np.float64).ravel()[valid]
        yp = y_pred.astype(np.float64).ravel()[valid]
    else:
        yt = y_true.astype(np.float64).ravel()
        yp = y_pred.astype(np.float64).ravel()

    mse = float(np.mean((yp - yt) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yp - yt)))
    pcc = float(np.corrcoef(yt, yp)[0, 1]) if yt.std() > 0 and yp.std() > 0 else 0.0
    mape = float(np.mean(np.abs((yp - yt) / np.maximum(np.abs(yt), 1e-6))) * 100.0)
    nrmse = float(rmse / (np.max(yt) - np.min(yt) + 1e-6))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / (np.sum((yt - yt.mean()) ** 2) + 1e-12))

    yt_img = y_true.squeeze(1).copy()
    yp_img = y_pred.squeeze(1).copy()
    if mask is not None:
        mask_img = mask.squeeze(1).astype(np.float64)
        yt_img = yt_img * mask_img
        yp_img = yp_img * mask_img
    data_range = float(max(yt_img.max(), yp_img.max()) - min(yt_img.min(), yp_img.min()) + 1e-6)
    ssim_vals = [
        structural_similarity(yt_img[i], yp_img[i], data_range=data_range)
        for i in range(yt_img.shape[0])
    ]

    return {
        "mse": mse,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "mape": mape,
        "pcc": pcc,
        "r2": r2,
        "ssim": float(np.mean(ssim_vals)),
    }


def round_metrics(metrics: dict[str, float], ndigits: int = 5) -> dict[str, float]:
    return {k: round(float(v), ndigits) for k, v in metrics.items()}
