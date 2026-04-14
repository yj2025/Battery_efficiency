import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

from battery_data import BatteryDataGenerator
from models import BatteryModelTrainer, BatteryPredictor
from visualization import BatteryPlotter

# 전역 무시 대신 특정 카테고리만 필터링
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── 타이밍 컨텍스트 ──────────────────────────────────────────────────
@contextmanager
def _timer(label: str):
    t0 = time.perf_counter()
    yield
    logger.info("%s: %.2fs", label, time.perf_counter() - t0)


# ── 테스트 케이스 설정 분리 ───────────────────────────────────────────
@dataclass
class TestCase:
    label:       str
    cycles:      int
    temperature: float
    c_rate:      float
    voltage:     float
    age_months:  int


TEST_CASES: list[TestCase] = [
    TestCase("새 배터리",    cycles=100,  temperature=25, c_rate=1.0, voltage=3.7, age_months=6),
    TestCase("중간 수명",    cycles=1000, temperature=25, c_rate=1.0, voltage=3.7, age_months=24),
    TestCase("노후 배터리",  cycles=1800, temperature=40, c_rate=2.0, voltage=3.5, age_months=48),
]


# ── 단계별 함수 분리 ──────────────────────────────────────────────────
def _generate_data(n_samples: int = 1000):
    logger.info("합성 데이터 생성 중... (%d 샘플)", n_samples)
    with _timer("데이터 생성"):
        generator = BatteryDataGenerator()
        return generator.generate_synthetic_data(n_samples)


def _train_models(data):
    logger.info("AI 모델 학습 중...")
    with _timer("모델 학습"):
        trainer = BatteryModelTrainer()
        metrics = trainer.train(data)

    logger.info("── 모델 성능 ──")
    for name, m in metrics.items():
        # ModelMetrics dataclass 속성 접근 (딕셔너리 아님)
        logger.info("  %-12s MAE=%.4f  R²=%.4f", name, m.mae, m.r2)

    return trainer


def _run_predictions(predictor) -> None:
    logger.info("── 성능 예측 예시 ──")
    for tc in TEST_CASES:
        result = predictor.predict(
            tc.cycles, tc.temperature, tc.c_rate, tc.voltage, tc.age_months
        )
        logger.info(
            "[%s] 용량=%.1f%%  효율=%.1f%%  전압=%.2fV  건강도=%.1f%%",
            tc.label,
            result["predicted_capacity"]  * 100,
            result["predicted_efficiency"] * 100,
            result["predicted_voltage"],
            result["health_score"],
        )


def _visualize(predictor, data) -> None:
    logger.info("결과 시각화 생성 중...")
    with _timer("시각화"):
        plotter = BatteryPlotter(predictor)
        plotter.plot(data, save_path="battery_analysis.png")   # 파일 저장 추가


# ── 메인 ──────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("🔋 배터리 성능 예측 시스템 시작")

    try:
        data    = _generate_data(n_samples=1000)
        trainer = _train_models(data)

        models    = trainer.get_models()
        predictor = BatteryPredictor(
            capacity_model=models["capacity"],
            efficiency_model=models["efficiency"],
            voltage_model=models["voltage_drop"],
        )

        _run_predictions(predictor)
        _visualize(predictor, data)

        logger.info("✅ 분석 완료")

    except ValueError as e:
        logger.error("입력 데이터 오류: %s", e)
        raise
    except RuntimeError as e:
        logger.error("모델 오류: %s", e)
        raise


if __name__ == "__main__":
    main()
