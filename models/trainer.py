from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FEATURES = ["cycles", "temperature", "c_rate", "voltage", "age_months"]
TARGETS  = ["capacity", "efficiency", "voltage_drop"]

# ── 하이퍼파라미터 한 곳에서 관리 ────────────────────────────────────
@dataclass
class RFConfig:
    n_estimators: int   = 100
    random_state: int   = 42
    test_size:    float = 0.2
    n_jobs:       int   = -1        # RF 내부 병렬
    max_workers:  int   = 3         # 모델 간 병렬


@dataclass
class ModelMetrics:
    mae: float
    r2:  float

    def __repr__(self) -> str:
        return f"MAE={self.mae:.4f}, R²={self.r2:.4f}"


# ── Trainer ───────────────────────────────────────────────────────────
class BatteryModelTrainer:
    def __init__(self, cfg: RFConfig | None = None) -> None:
        self.cfg    = cfg or RFConfig()
        self._models: dict[str, RandomForestRegressor] = {}   # 학습 전엔 비어있음

    # ── 내부: 단일 모델 학습 (thread-safe) ───────────────────────────
    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        X_test:  pd.DataFrame,
        y_train: pd.Series,
        y_test:  pd.Series,
        target:  str,
    ) -> tuple[str, RandomForestRegressor, ModelMetrics]:

        model = RandomForestRegressor(
            n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
        )
        model.fit(X_train, y_train)

        pred    = model.predict(X_test)
        metrics = ModelMetrics(
            mae=mean_absolute_error(y_test, pred),
            r2 =r2_score(y_test, pred),
        )
        logger.info("[%s] %s", target, metrics)
        return target, model, metrics

    # ── 공개: 학습 ───────────────────────────────────────────────────
    def train(self, data: pd.DataFrame) -> dict[str, ModelMetrics]:
        """세 타겟을 병렬로 학습하고 metrics를 반환합니다."""
        missing = set(FEATURES + TARGETS) - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame에 필요한 컬럼이 없습니다: {missing}")

        X = data[FEATURES]

        # split은 X 기준으로 한 번만 → 세 모델이 동일한 test set 공유
        X_train, X_test, idx_train, idx_test = train_test_split(
            X, X.index,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
        )

        all_metrics: dict[str, ModelMetrics] = {}

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
            futures = {
                pool.submit(
                    self._train_single_model,
                    X_train, X_test,
                    data.loc[idx_train, target],
                    data.loc[idx_test,  target],
                    target,
                ): target
                for target in TARGETS
            }
            for future in as_completed(futures):
                target, model, metrics = future.result()   # 예외도 여기서 전파
                self._models[target]   = model
                all_metrics[target]    = metrics

        return all_metrics

    # ── 공개: 모델 반환 (미학습 시 명시적 에러) ──────────────────────
    def get_models(self) -> dict[str, RandomForestRegressor]:
        if len(self._models) != len(TARGETS):
            missing = set(TARGETS) - set(self._models)
            raise RuntimeError(f"아직 학습되지 않은 모델: {missing}")
        return dict(self._models)   # 얕은 복사로 내부 dict 보호
