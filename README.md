# 🔋 배터리 성능 예측 시스템 (Battery Performance Predictor)

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

| 파라미터 | 설명 | 범위 | 단위 |
|---------|------|------|------|
| cycles | 충방전 사이클 수 | 1-2000 | 회 |
| temperature | 동작 온도 | -10~60 | °C |
| c_rate | 충방전율 | 0.1-3.0 | C |
| voltage | 공칭 전압 | 3.0-4.2 | V |
| age_months | 사용 기간 | 0-60 | 개월 |

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
