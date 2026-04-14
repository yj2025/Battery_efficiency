# 🔋 배터리 성능 예측 시스템

<<<<<<< HEAD
Random Forest 기반으로 배터리의 용량, 효율성, 전압 특성을 예측하는 분석 도구입니다.  
물리학 기반 합성 데이터(Arrhenius 열화 모델 + 캘린더 에이징)로 학습하며, 다양한 사용 조건에서의 배터리 상태를 수치와 그래프로 확인할 수 있습니다.

---

## 프로젝트 구조

```
BATTERY_EFFICIENCY/
├── battery_data/       # 합성 데이터 생성
├── models/
│   ├── trainer.py      # Random Forest 학습 (3개 타겟 병렬 학습)
│   └── predictor.py    # 단일 / 배치 예측
├── visualization/
│   └── plotter.py      # 6종 분석 그래프
└── main.py
```

---

## 예측 결과

| 출력값 | 설명 | 범위 |
|--------|------|------|
| `predicted_capacity` | 초기 대비 현재 용량 비율 | 0.0 – 1.0 |
| `predicted_efficiency` | 에너지 효율 | 0.70 – 0.98 |
| `predicted_voltage` | 부하 조건에서의 실제 전압 | V |
| `health_score` | 종합 건강도 | 0 – 100% |

건강도 기준은 90% 이상이면 우수, 80–90%는 양호, 70–80%는 교체 권장, 70% 미만은 교체 필수입니다.

---

## 입력 파라미터

| 파라미터 | 설명 | 허용 범위 |
|----------|------|-----------|
| `cycles` | 충방전 사이클 수 | 0 – 10,000 |
| `temperature` | 동작 온도 (°C) | -40 – 85 |
| `c_rate` | 충방전율 | 0.0 – 10.0 |
| `voltage` | 공칭 전압 (V) | 0.0 – 5.0 |
| `age_months` | 사용 기간 (개월) | 0 – 240 |

---

## 사용법

단일 예측:

```python
result = predictor.predict(
    cycles=1000, temperature=25, c_rate=1.0, voltage=3.7, age_months=24
)
print(result["health_score"])  # ex) 82.4
```

배치 예측:

```python
import numpy as np

data = np.array([
    [100,  25, 1.0, 3.7,  6],
    [1000, 35, 1.5, 3.6, 24],
    [1800, 45, 2.0, 3.4, 48],
])
results = predictor.predict_batch(data)
```

---
=======
Random Forest 알고리즘과 물리학 기반 모델링을 통해 배터리의 용량, 효율성, 전압 특성을 예측하는 시스템입니다.

## 주요 기능

- 충방전 사이클, 온도, C-rate 등을 고려한 배터리 성능 예측
- 시간에 따른 배터리 건강도 변화 시뮬레이션
- 온도와 사용 조건이 성능에 미치는 영향 분석
- 6가지 그래프를 통한 종합적인 성능 시각화

## 기술적 특징

**물리학 기반 모델링**
- 아레니우스 방정식을 활용한 온도별 열화 모델링
- 지수 감소 모델 기반 용량 감소: `Q(t) = Q₀ × exp(-kt)`
- 캘린더 에이징(시간 경과에 따른 자연 열화) 반영

**AI 예측 모델**
- Random Forest로 용량, 효율성, 전압을 동시 예측
- 특성 중요도 분석을 통한 요인별 영향력 정량화

## 입력 파라미터
>>>>>>> 80d774aacf3ce7a1f1db3c1edfa84e7ad679aab8

## 시각화

<<<<<<< HEAD
`plotter.plot(data)` 호출 시 6종 그래프를 한 번에 출력합니다.

1. **사이클 vs 용량** — 충방전 반복에 따른 용량 감소 추이
2. **온도 vs 효율성** — 온도 변화가 효율에 미치는 영향
3. **C-rate vs 전압** — 방전율에 따른 전압 강하 특성
4. **피처 중요도** — 각 입력 변수가 용량 예측에 기여하는 비중
5. **건강도 추이** — 사용 기간에 따른 건강도 변화
6. **3D 효율 맵** — 온도 × C-rate 조합에 따른 효율성 지형도

결과 이미지는 `battery_analysis.png`로 자동 저장됩니다.

---

## 모델 구조

세 가지 타겟(용량, 효율성, 전압 강하)을 각각 별도의 Random Forest로 학습하며, ThreadPoolExecutor로 병렬 처리합니다.

```
용량 감소 = 0.4 × cycle_factor + 0.3 × temp_factor + 0.3 × age_factor

cycle_factor = 1 - (cycles / 5000) × (1 + c_rate × 0.2)
temp_factor  = Arrhenius 정규화 (Ea = 0.06 eV)
age_factor   = exp(-age_months / 120)
```

모델 평가 지표는 MAE와 R²를 사용하며, 학습 완료 후 콘솔에 출력됩니다.

---

## 한계 및 주의사항

- 합성 데이터 기반이므로 실제 배터리와 특성 차이가 있을 수 있습니다.
- 학습 데이터 분포 특성상 극한 온도(-40°C, 80°C 이상) 영역의 예측 신뢰도가 낮습니다.
- 배터리 화학적 조성(리튬이온, LFP 등) 차이는 반영되지 않습니다.
- 실사용 환경에 적용하려면 실측 데이터로 재학습을 권장합니다.

---

## 요구사항

- Python 3.7+
- scikit-learn, numpy, pandas, matplotlib
=======
## 출력값 설명

- `predicted_capacity`: 초기 용량 대비 현재 용량 비율 (0.0-1.0)
- `predicted_efficiency`: 에너지 효율성 (0.7-0.98)
- `predicted_voltage`: 부하 조건에서의 실제 전압 (V)
- `health_score`: 종합 배터리 건강도 (0-100%)

건강도 기준: 90% 이상 우수 / 80-90% 양호 / 70-80% 교체 권장 / 70% 미만 교체 필수

## 시각화 구성

1. 사이클 vs 용량: 충방전 반복에 따른 용량 감소 추이
2. 온도 vs 효율성: 온도 변화가 효율에 미치는 영향
3. C-rate vs 전압: 방전율에 따른 전압 강하 특성
4. 특성 중요도: 각 요인이 용량 예측에 미치는 영향력
5. 배터리 수명: 시간에 따른 건강도 변화 시뮬레이션
6. 3D 성능 맵: 온도-C-rate-효율성 관계 3차원 시각화

## 사용법

**기본 예측**
```python
result = predictor.predict_performance(cycles, temperature, c_rate, voltage, age_months)
print(f"건강도: {result['health_score']:.1f}%")
```

**커스텀 데이터로 재학습**
```python
import pandas as pd
your_data = pd.read_csv('battery_data.csv')
predictor.train_models(your_data)
```

**배치 예측**
```python
conditions = [
    (100, 25, 1.0, 3.7, 6),
    (1000, 35, 1.5, 3.6, 24),
    (1800, 45, 2.0, 3.4, 48)
]
for condition in conditions:
    result = predictor.predict_performance(*condition)
    print(f"건강도: {result['health_score']:.1f}%")
```

## 모델 성능

- 용량 예측: R² > 0.90
- 효율성 예측: R² > 0.85
- 전압 예측: R² > 0.80

## 한계 및 주의사항

- 합성 데이터 기반이므로 실제 배터리와 차이가 있을 수 있음
- 극한 온도 조건에서 정확도 저하 가능
- 배터리 화학적 조성 차이 미고려
- 실사용 시 실측 데이터로 재학습 권장

## 시스템 요구사항

- Python 3.7 이상
- RAM 4GB 이상
- 저장공간 100MB 이상

## License

MIT
>>>>>>> 80d774aacf3ce7a1f1db3c1edfa84e7ad679aab8
