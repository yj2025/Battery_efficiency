import numpy as np
import pandas as pd


class BatteryDataGenerator:
    def generate_synthetic_data(self, n_samples=1000):
        """
        실제 배터리 데이터를 모방한 합성 데이터 생성
        특징: 충방전 사이클, 온도, C-rate, 전압 등
        """
        np.random.seed(42)

        cycles      = np.random.randint(1, 2000, n_samples)
        temperature = np.random.normal(25, 10, n_samples)
        c_rate      = np.random.uniform(0.1, 3.0, n_samples)
        voltage     = np.random.uniform(3.0, 4.2, n_samples)
        age_months  = np.random.uniform(0, 60, n_samples)

        # ── 용량 (capacity) ───────────────────────────────────────────
        # 각 인자를 독립적으로 계산 후 가중 평균으로 균등하게 반영

        # 사이클: 많을수록 용량 감소 (c_rate가 높으면 더 빨리 감소)
        cycle_factor = 1 - (cycles / 5000) * (1 + c_rate * 0.2)

        # 온도: Arrhenius activation_energy 축소로 스케일 완화
        temp_kelvin       = temperature + 273.15
        activation_energy = 0.06          # 0.6 → 0.06 (10배 축소)
        k_B               = 8.617e-5
        k_temp            = np.exp(-activation_energy / (k_B * temp_kelvin))
        temp_factor       = k_temp / np.max(k_temp)

        # 노화: 시간이 지날수록 완만하게 감소
        age_factor = np.exp(-age_months / 120)

        # 가중 평균으로 균등하게 합산 (cycle 40%, temp 30%, age 30%)
        capacity = np.clip(
            0.4 * cycle_factor +
            0.3 * temp_factor  +
            0.3 * age_factor   +
            np.random.normal(0, 0.02, n_samples),
            0.3, 1.0
        )

        # ── 효율성 (efficiency) ───────────────────────────────────────
        base_efficiency = 0.95
        temp_penalty    = np.abs(temperature - 25) * 0.001
        c_rate_penalty  = (c_rate - 1.0) ** 2 * 0.05

        efficiency = np.clip(
            base_efficiency - temp_penalty - c_rate_penalty +
            np.random.normal(0, 0.01, n_samples),
            0.7, 0.98
        )

        # ── 전압 강하 (voltage_drop) ──────────────────────────────────
        resistance_increase = (1 - capacity) * 0.5
        voltage_drop        = voltage * (1 - resistance_increase * c_rate * 0.1)

        return pd.DataFrame({
            "cycles":      cycles,
            "temperature": temperature,
            "c_rate":      c_rate,
            "voltage":     voltage,
            "age_months":  age_months,
            "capacity":    capacity,
            "efficiency":  efficiency,
            "voltage_drop": voltage_drop,
        })
