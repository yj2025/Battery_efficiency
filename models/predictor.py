import numpy as np
from numpy.typing import ArrayLike


class BatteryPredictor:
    # 물리적 허용 범위 상수화
    _VALID_RANGES = {
        "cycles":       (0,    10_000),
        "temperature":  (-40,  85),
        "c_rate":       (0.0,  10.0),
        "voltage":      (0.0,  5.0),
        "age_months":   (0,    240),
    }

    def __init__(self, capacity_model, efficiency_model, voltage_model):
        self.capacity_model   = capacity_model
        self.efficiency_model = efficiency_model
        self.voltage_model    = voltage_model

    # ── 입력 검증 (재사용 가능하도록 분리) ──────────────────────────
    def _validate(
        self,
        cycles: float,
        temperature: float,
        c_rate: float,
        voltage: float,
        age_months: float,
    ) -> None:
        values = dict(
            cycles=cycles,
            temperature=temperature,
            c_rate=c_rate,
            voltage=voltage,
            age_months=age_months,
        )
        for name, val in values.items():
            lo, hi = self._VALID_RANGES[name]
            if not (lo <= val <= hi):
                raise ValueError(
                    f"'{name}' = {val} is out of valid range [{lo}, {hi}]"
                )

    # ── 단일 샘플 예측 ───────────────────────────────────────────────
    def predict(
        self,
        cycles: float,
        temperature: float,
        c_rate: float,
        voltage: float,
        age_months: float,
    ) -> dict[str, float]:
        self._validate(cycles, temperature, c_rate, voltage, age_months)

        # 배열 한 번만 생성 → 세 모델이 공유
        X = np.array([[cycles, temperature, c_rate, voltage, age_months]],
                     dtype=np.float32)

        capacity     = float(self.capacity_model.predict(X)[0])
        efficiency   = float(self.efficiency_model.predict(X)[0])
        voltage_drop = float(self.voltage_model.predict(X)[0])

        # capacity·efficiency를 명시적으로 [0,1] 정규화 후 점수화
        health_score = np.clip((capacity + efficiency) / 2, 0.0, 1.0) * 100

        return {
            "predicted_capacity":   capacity,
            "predicted_efficiency": efficiency,
            "predicted_voltage":    voltage_drop,
            "health_score":         float(health_score),
        }

    # ── 배치 예측 (루프 없이 벡터화) ────────────────────────────────
    def predict_batch(self, data: ArrayLike) -> list[dict[str, float]]:
        """
        data : shape (N, 5) — [cycles, temperature, c_rate, voltage, age_months]
        """
        X = np.asarray(data, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != 5:
            raise ValueError(f"data must be shape (N, 5), got {X.shape}")

        capacities   = self.capacity_model.predict(X)
        efficiencies = self.efficiency_model.predict(X)
        voltages     = self.voltage_model.predict(X)
        health_scores = np.clip((capacities + efficiencies) / 2, 0.0, 1.0) * 100

        return [
            {
                "predicted_capacity":   float(c),
                "predicted_efficiency": float(e),
                "predicted_voltage":    float(v),
                "health_score":         float(h),
            }
            for c, e, v, h in zip(capacities, efficiencies, voltages, health_scores)
        ]
