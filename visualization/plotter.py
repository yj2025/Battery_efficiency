from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_DEFAULT_FIXED = dict(cycles=1000, temperature=25.0, c_rate=1.0, voltage=3.7, age_months=12)


class BatteryPlotter:
    """배터리 성능 예측 결과 시각화 (2D 그래프 + 3D 효율 맵)"""

    _SCATTER_KWARGS = dict(alpha=0.5, s=20)

    def __init__(self, predictor, scatter_sample: int = 100) -> None:
        self.predictor      = predictor
        self.scatter_sample = scatter_sample

    # ── 공개 진입점 ───────────────────────────────────────────────────
    def plot(
        self,
        data: pd.DataFrame,
        *,
        save_path: str | None = None,
        show: bool = True,
    ) -> plt.Figure:
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle("배터리 성능 예측 시스템 분석", fontsize=16, fontweight="bold")

        axes_2d = [fig.add_subplot(2, 3, i) for i in range(1, 6)]
        ax_3d   = fig.add_subplot(2, 3, 6, projection="3d")

        self._plot_cycle_vs_capacity(axes_2d[0], data)
        self._plot_temp_vs_efficiency(axes_2d[1], data)
        self._plot_crate_vs_voltage(axes_2d[2], data)
        self._plot_feature_importance(axes_2d[3])
        self._plot_health_over_time(axes_2d[4])
        self._plot_3d_efficiency_map(ax_3d)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── 내부 유틸 ─────────────────────────────────────────────────────
    @staticmethod
    def _style_ax(ax, xlabel: str, ylabel: str, title: str) -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _batch_predict(
        self,
        key: str,
        fixed: dict,
        sweep_param: str,
        sweep_values: np.ndarray,
    ) -> np.ndarray:
        """sweep_param만 바꿔가며 배치 예측 (predict_batch 활용)"""
        rows = [{**fixed, sweep_param: v} for v in sweep_values]
        data = np.array([
            [r["cycles"], r["temperature"], r["c_rate"], r["voltage"], r["age_months"]]
            for r in rows
        ], dtype=np.float32)

        results = self.predictor.predict_batch(data)
        return np.array([r[key] for r in results])

    # ── 2D 그래프 ─────────────────────────────────────────────────────
    def _plot_cycle_vs_capacity(self, ax, data: pd.DataFrame) -> None:
        cycles_range = np.linspace(1, 2000, 100)
        fixed        = {k: v for k, v in _DEFAULT_FIXED.items() if k != "cycles"}

        predictions = self._batch_predict("predicted_capacity", fixed, "cycles", cycles_range)

        ax.plot(cycles_range, predictions, "b-", linewidth=2, label="예측값")
        sample = data.iloc[:self.scatter_sample]
        ax.scatter(
            sample["cycles"], sample["capacity"],
            c="red", label="실제값", **self._SCATTER_KWARGS,
        )
        self._style_ax(ax, "충방전 사이클", "용량 비율", "사이클에 따른 용량 감소")

    def _plot_temp_vs_efficiency(self, ax, data: pd.DataFrame) -> None:
        temp_range = np.linspace(-40, 85, 100)
        fixed      = {k: v for k, v in _DEFAULT_FIXED.items() if k != "temperature"}

        predictions = self._batch_predict("predicted_efficiency", fixed, "temperature", temp_range)

        ax.plot(temp_range, predictions, "g-", linewidth=2, label="예측값")
        sample = data.iloc[:self.scatter_sample]
        ax.scatter(
            sample["temperature"], sample["efficiency"],
            c="red", label="실제값", **self._SCATTER_KWARGS,
        )
        self._style_ax(ax, "온도 (°C)", "효율성", "온도에 따른 효율성 변화")

    def _plot_crate_vs_voltage(self, ax, data: pd.DataFrame) -> None:
        crate_range = np.linspace(0.5, 2.5, 100)
        fixed       = {k: v for k, v in _DEFAULT_FIXED.items() if k != "c_rate"}

        predictions = self._batch_predict("predicted_voltage", fixed, "c_rate", crate_range)

        ax.plot(crate_range, predictions, "r-", linewidth=2, label="예측값")
        sample = data.iloc[:self.scatter_sample]
        ax.scatter(
            sample["c_rate"], sample["voltage_drop"],
            c="blue", label="실제값", **self._SCATTER_KWARGS,
        )
        self._style_ax(ax, "C-rate", "전압 강하 (V)", "C-rate에 따른 전압 변화")

    def _plot_feature_importance(self, ax) -> None:
        importance = self.predictor.capacity_model.feature_importances_
        features   = ["cycles", "temperature", "c_rate", "voltage", "age_months"]

        ax.barh(features, importance, color="steelblue")
        ax.set_xlabel("중요도")
        ax.set_title("용량 예측 피처 중요도")
        ax.grid(True, alpha=0.3, axis="x")

    def _plot_health_over_time(self, ax) -> None:
        months_range = np.linspace(0, 60, 100)
        fixed        = {k: v for k, v in _DEFAULT_FIXED.items() if k != "age_months"}

        predictions = self._batch_predict("health_score", fixed, "age_months", months_range)

        ax.plot(months_range, predictions, color="purple", linewidth=2, label="건강도")
        ax.set_ylim(0, 100)
        self._style_ax(ax, "사용 개월 수", "건강도 (%)", "시간에 따른 배터리 건강도")

    # ── 3D 효율 맵 ────────────────────────────────────────────────────
    def _plot_3d_efficiency_map(self, ax_3d, n_grid: int = 20) -> None:
        temp_1d  = np.linspace(0, 50, n_grid)
        crate_1d = np.linspace(0.5, 2.5, n_grid)
        temp_mesh, crate_mesh = np.meshgrid(temp_1d, crate_1d)

        flat_temps  = temp_mesh.ravel()
        flat_crates = crate_mesh.ravel()
        batch_input = np.column_stack([
            np.full_like(flat_temps, 1000),
            flat_temps,
            flat_crates,
            np.full_like(flat_temps, 3.7),
            np.full_like(flat_temps, 24),
        ]).astype(np.float32)

        results         = self.predictor.predict_batch(batch_input)
        efficiency_flat = np.array([r["predicted_efficiency"] for r in results])
        efficiency_mesh = efficiency_flat.reshape(temp_mesh.shape)

        ax_3d.plot_surface(temp_mesh, crate_mesh, efficiency_mesh, cmap="viridis", alpha=0.8)
        ax_3d.set_xlabel("온도 (°C)")
        ax_3d.set_ylabel("C-rate")
        ax_3d.set_zlabel("효율성")
        ax_3d.set_title("온도-C-rate 효율성 맵")