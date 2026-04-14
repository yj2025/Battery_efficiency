import logging
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

from battery_data import BatteryDataGenerator
from models import BatteryModelTrainer, BatteryPredictor
from visualization import BatteryPlotter

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@contextmanager
def _timer(label: str):
    """블록 실행 시간을 측정해 로그로 출력하는 컨텍스트 매니저."""
    t0 = time.perf_counter()
    yield
    logger.info("%s: %.2fs", label, time.perf_counter() - t0)


@dataclass
class TestCase:
    """단일 예측 시나리오를 정의하는 설정값 묶음."""
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
    TestCase("고온 환경",    cycles=500,  temperature=45, c_rate=1.5, voltage=3.6, age_months=12),
]


def _generate_data(n_samples: int = 1000):
    """물리학 기반 합성 배터리 데이터를 생성한다."""
    logger.info("합성 데이터 생성 중... (%d 샘플)", n_samples)
    with _timer("데이터 생성"):
        generator = BatteryDataGenerator()
        return generator.generate_synthetic_data(n_samples)


def _train_models(data):
    """Random Forest 모델 3개(용량, 효율성, 전압)를 병렬 학습한다."""
    logger.info("AI 모델 학습 중...")
    with _timer("모델 학습"):
        trainer = BatteryModelTrainer()
        metrics = trainer.train(data)

    logger.info("── 모델 성능 ──")
    for name, m in metrics.items():
        logger.info("  %-12s MAE=%.4f  R²=%.4f", name, m.mae, m.r2)

    return trainer


def _run_predictions(predictor) -> list[dict]:
    """TEST_CASES 각 시나리오에 대해 예측을 실행하고 결과를 반환한다."""
    logger.info("── 성능 예측 예시 ──")
    results = []
    for tc in TEST_CASES:
        result = predictor.predict(
            tc.cycles, tc.temperature, tc.c_rate, tc.voltage, tc.age_months
        )
        results.append({"label": tc.label, **result})
        logger.info(
            "[%s] 용량=%.1f%%  효율=%.1f%%  전압=%.2fV  건강도=%.1f%%",
            tc.label,
            result["predicted_capacity"]  * 100,
            result["predicted_efficiency"] * 100,
            result["predicted_voltage"],
            result["health_score"],
        )
    return results


def _visualize(predictor, data) -> None:
    """분석 그래프 6종을 생성하고 파일로 저장한다."""
    logger.info("결과 시각화 생성 중...")
    with _timer("시각화"):
        plotter = BatteryPlotter(predictor)
        plotter.plot(data, save_path="battery_analysis.png")


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

        results = _run_predictions(predictor)  # 추후 그래프 연동 시 _visualize에 전달
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