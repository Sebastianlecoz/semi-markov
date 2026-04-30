
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from rich.console import Console
console = Console()

class AFTLibrary:
    """
    Wrapper around fitted lifelines AFT models.

    Key fix: at prediction time, covariates are scaled using the SAME
    StandardScaler that was fitted during training and saved to
    models/aft/feature_scaler.pkl.  Previously, manual approximations
    (dist/120, etc.) were used, which produced out-of-distribution values
    and made disp_scaled effectively invisible to the models.
    """

    # Paths to search for the scaler (tried in order)
    _SCALER_SEARCH_PATHS = [
        Path("models/aft/feature_scaler.pkl"),
        Path("../../../models/aft/feature_scaler.pkl"),
        Path("/home/firasboustila/smc/models/aft/feature_scaler.pkl"),
    ]

    def __init__(self, aft_models: Dict[str, Any], scaler=None):
        self.models = aft_models or {}
        self._lookup_cache: Dict[tuple, Any] = {}

        if scaler is not None:
            self._scaler = scaler
            console.print("[green]✓ feature_scaler injected directly[/green]")
        else:
            self._scaler = self._load_scaler()
            if self._scaler is None:
                console.print(
                    "[yellow]⚠ feature_scaler.pkl not found — "
                    "falling back to manual approximations (less accurate)[/yellow]"
                )
            else:
                console.print("[green]✓ feature_scaler.pkl loaded[/green]")
    def _load_scaler(self) -> Optional[Any]:
        for path in self._SCALER_SEARCH_PATHS:
            if path.exists():
                try:
                    return joblib.load(path)
                except Exception as e:
                    console.print(f"[yellow]Could not load scaler from {path}: {e}[/yellow]")
        return None

    # ------------------------------------------------------------------
    # Key lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _key(from_state: str, to_state: str) -> List[str]:
        return [
            f"{from_state}_->_{to_state}",
            f"{from_state}_-_{to_state}",
            f"{from_state} -> {to_state}",
            f"{from_state}_{to_state}",
        ]

    def _find(self, from_state: str, to_state: str) -> Optional[Any]:
        cache_key = (from_state, to_state)
        if cache_key in self._lookup_cache:
            return self._lookup_cache[cache_key]
        for name in self._key(from_state, to_state):
            if name in self.models:
                self._lookup_cache[cache_key] = self.models[name]
                return self.models[name]
        return None

    # ------------------------------------------------------------------
    # Covariate construction — THE critical fix
    # ------------------------------------------------------------------

    def _build_X(
            self,
            dist: float,
            ang: float,
            disp: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Build the covariate DataFrame using the REAL StandardScaler.
        Passes a named DataFrame to scaler.transform() to avoid sklearn's
        'X does not have valid feature names' warning.
        """
        if self._scaler is not None:
            if disp is not None:
                raw = pd.DataFrame(
                    {"dist": [dist], "ang": [ang], "disp": [disp]}
                )
                scaled = self._scaler.transform(raw)[0]
                return pd.DataFrame({
                    "dist_scaled": [float(scaled[0])],
                    "ang_scaled":  [float(scaled[1])],
                    "disp_scaled": [float(scaled[2])],
                    "Intercept":   [1.0],
                })
            else:
                raw = pd.DataFrame({"dist": [dist], "ang": [ang]})
                # Scaler was fitted on 3 features; transform only the first two.
                import numpy as np
                mean  = self._scaler.mean_[:2]
                scale = self._scaler.scale_[:2]
                dist_s = (dist - mean[0]) / scale[0]
                ang_s  = (ang  - mean[1]) / scale[1]
                return pd.DataFrame({
                    "dist_scaled": [float(dist_s)],
                    "ang_scaled":  [float(ang_s)],
                    "Intercept":   [1.0],
                })
        else:
            # Fallback: manual approximations (no scaler available)
            import numpy as np
            row = {
                "dist_scaled": [float(np.clip(dist / 120.0, 0.01, 0.99))],
                "ang_scaled":  [float(np.clip((ang + np.pi) / (2 * np.pi), 0.01, 0.99))],
                "Intercept":   [1.0],
            }
            if disp is not None:
                row["disp_scaled"] = [float(np.clip(disp / 120.0, 0.0, 1.0))]
            return pd.DataFrame(row)

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------

    @staticmethod
    def _model_uses_disp(model: Any) -> bool:
        """Return True if the fitted model has disp_scaled as a covariate."""
        if hasattr(model, "summary"):
            try:
                idx = model.summary.index
                covariates = (
                    idx.get_level_values(-1).tolist()
                    if hasattr(idx, "levels")
                    else idx.tolist()
                )
                return "disp_scaled" in covariates
            except Exception:
                pass
        if hasattr(model, "feature_names_in_"):
            return "disp_scaled" in model.feature_names_in_
        return False

    # ------------------------------------------------------------------
    # Survival / mass helpers
    # ------------------------------------------------------------------

    def survival_at_times(
            self, model: Any, times: np.ndarray, X: pd.DataFrame
    ) -> np.ndarray:
        sf = model.predict_survival_function(X, times=times)
        return np.clip(sf.values.flatten(), 1e-6, 1.0)

    def mass_in_bin(
            self,
            from_state: str,
            to_state: str,
            dist: float,
            ang: float,
            t_hi: float,
            delta_t: float,
            disp: Optional[float] = None,
    ) -> float:
        if t_hi <= 0 or delta_t <= 0:
            return 0.0
        model = self._find(from_state, to_state)
        if model is None:
            return 0.0
        use_disp = self._model_uses_disp(model)
        X = self._build_X(dist, ang, disp if use_disp else None)
        t_lo = max(t_hi - delta_t, 0.01)
        S = self.survival_at_times(model, np.array([t_lo, t_hi]), X)
        return float(np.clip((1.0 - S[1]) - (1.0 - S[0]), 0.0, 1.0))

    def get_mass_distribution(
            self,
            from_state: str,
            to_state: str,
            dist: float,
            ang: float,
            k_max: int,
            delta_t: float,
            disp: Optional[float] = None,
    ) -> np.ndarray:
        """
        Return all k_max bin masses in ONE model prediction call.

        disp (metres) is the actual displacement to pass to the model.
        Pass None to omit disp_scaled (model uses dist+ang only).
        """
        model = self._find(from_state, to_state)
        if model is None:
            return np.zeros(k_max)
        use_disp = self._model_uses_disp(model)
        X = self._build_X(dist, ang, disp if use_disp else None)
        time_points = np.linspace(1e-6, k_max * delta_t, k_max + 1)
        S = self.survival_at_times(model, time_points, X)
        masses = np.diff(1.0 - S)
        return np.clip(masses, 0.0, 1.0)

