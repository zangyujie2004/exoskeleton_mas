from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:  # pragma: no cover
    RandomForestRegressor = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover
    XGBRegressor = None

SKLEARN_MODEL_TYPES: frozenset[str] = frozenset({"random_forest", "rf", "xgboost", "xgb"})


def is_sklearn_model_type(model_type: str) -> bool:
    return str(model_type).strip().lower() in SKLEARN_MODEL_TYPES


class SklearnRegressorModule(nn.Module):
    """Wraps sklearn / XGBoost regressor for tabular flattened EMG (B, C*T) -> scalar."""

    def __init__(self, estimator: Any, name: str) -> None:
        super().__init__()
        self.estimator = estimator
        self.name = name
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> SklearnRegressorModule:
        self.estimator.fit(X, y.astype(np.float64, copy=False))
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("SklearnRegressorModule: call fit() before forward()")
        if x.ndim == 3:
            x = x.reshape(x.size(0), -1)
        xp = x.detach().cpu().numpy().astype(np.float64, copy=False)
        y = self.estimator.predict(xp).astype(np.float32, copy=False)
        return torch.from_numpy(y).to(device=x.device, dtype=torch.float32)

    def state_dict(self, destination: Any = None, prefix: str = "", keep_vars: bool = False) -> dict[str, Any]:
        # 非 torch 参数；真实持久化用 checkpoint 里的 estimator
        return {}

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ) -> Any:  # noqa: ANN401
        return None


def dataloader_to_xy_arrays(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """(N, C*T) float64, (N,) float32 from batches of (B, C, T), (B,)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for x, y in loader:
        if x.ndim != 3:
            raise ValueError(f"Expected x (B, C, T), got {tuple(x.shape)}")
        b = x.size(0)
        xs.append(x.reshape(b, -1).numpy().astype(np.float64, copy=False))
        ys.append(y.detach().numpy().reshape(-1).astype(np.float32, copy=False))
    if not xs:
        return np.zeros((0, 1), dtype=np.float64), np.zeros((0,), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def build_random_forest_regressor(
    *,
    random_state: int = 42,
    n_estimators: int = 50,
    max_depth: int | None = 3,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    n_jobs: int = -1,
) -> SklearnRegressorModule:
    if RandomForestRegressor is None:
        raise ImportError("需要安装 scikit-learn: pip install scikit-learn")
    est = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    return SklearnRegressorModule(est, "random_forest")


def build_xgboost_regressor(
    *,
    random_state: int = 42,
    n_estimators: int = 10,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    n_jobs: int = -1,
) -> SklearnRegressorModule:
    if XGBRegressor is None:
        raise ImportError("需要安装 xgboost: pip install xgboost")
    est = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
        verbosity=0,
    )
    return SklearnRegressorModule(est, "xgboost")


def build_sklearn_from_cfg(model_type: str, cfg: dict[str, Any], *, random_state: int) -> SklearnRegressorModule:
    mt = str(model_type).strip().lower()
    if mt in ("random_forest", "rf"):
        sub = cfg.get("ml_random_forest")
        p: dict[str, Any] = {
            "random_state": random_state,
            "n_estimators": 300,
            "max_depth": 16,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "n_jobs": -1,
        }
        if isinstance(sub, dict):
            for k in _rf_kw():
                if k in sub:
                    p[k] = sub[k]
        return build_random_forest_regressor(**p)
    if mt in ("xgboost", "xgb"):
        sub = cfg.get("ml_xgboost")
        p = {
            "random_state": random_state,
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "n_jobs": -1,
        }
        if isinstance(sub, dict):
            for k in _xgb_kw():
                if k in sub:
                    p[k] = sub[k]
        return build_xgboost_regressor(**p)
    raise ValueError(f"Unknown sklearn model_type: {model_type!r}")


def _rf_kw() -> set[str]:
    return {
        "random_state",
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
        "n_jobs",
    }


def _xgb_kw() -> set[str]:
    return {
        "random_state",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "n_jobs",
    }
