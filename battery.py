import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class BatteryPerformancePredictor:
    def __init__(self):
        self.capacity_model = None
        self.efficiency_model = None
        self.voltage_model = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        실제 배터리 데이터를 모방한 합성 데이터 생성
        특징: 충방전 사이클, 온도, C-rate, 전압 등
        """
        np.random.seed(42)
        
        # 입력 특징들
        cycles = np.random.randint(1, 2000, n_samples)
        temperature = np.random.normal(25, 10, n_samples)  # 섭씨
        c_rate = np.random.uniform(0.1, 3.0, n_samples)  # C-rate
        voltage = np.random.uniform(3.0, 4.2, n_samples)  # 전압 (V)
        age_months = np.random.uniform(0, 60, n_samples)  # 사용 개월수
        
        # 물리학 기반 용량 감소 모델 (Arrhenius 방정식 포함)
        # Q(t) = Q0 * exp(-k*t) 형태의 지수 감소
        temp_kelvin = temperature + 273.15
        activation_energy = 0.6  # eV
        k_B = 8.617e-5  # 볼츠만 상수 (eV/K)
        
        # 온도 의존성 (아레니우스 방정식)
        k_temp = np.exp(-activation_energy / (k_B * temp_kelvin))
        
        # 사이클 및 C-rate 영향
        cycle_factor = 1 - (cycles / 5000) * (1 + c_rate * 0.2)
        temp_factor = k_temp / np.max(k_temp)
        age_factor = np.exp(-age_months / 120)  # 캘린더 에이징
        
        # 용량 (초기 용량 대비 비율)
        capacity = np.clip(
            cycle_factor * temp_factor * age_factor + 
            np.random.normal(0, 0.02, n_samples), 0.5, 1.0
        )
        
        # 효율성 (에너지 효율)
        base_efficiency = 0.95
        temp_penalty = np.abs(temperature - 25) * 0.001
        c_rate_penalty = (c_rate - 1.0) ** 2 * 0.05
        
        efficiency = np.clip(
            base_efficiency - temp_penalty - c_rate_penalty + 
            np.random.normal(0, 0.01, n_samples), 0.7, 0.98
        )
        
        # 전압 강하 (내부 저항 증가로 인한)
        resistance_increase = (1 - capacity) * 0.5
        voltage_drop = voltage * (1 - resistance_increase * c_rate * 0.1)
        
        return pd.DataFrame({
            'cycles': cycles,
            'temperature': temperature,
            'c_rate': c_rate,
            'voltage': voltage,
            'age_months': age_months,
            'capacity': capacity,
            'efficiency': efficiency,
            'voltage_drop': voltage_drop
        })
    
    def train_models(self, data):
        """
        Random Forest를 사용한 모델 학습
        """
        features = ['cycles', 'temperature', 'c_rate', 'voltage', 'age_months']
        X = data[features]
        
        # 용량 예측 모델
        y_capacity = data['capacity']
        X_train, X_test, y_train, y_test = train_test_split(X, y_capacity, test_size=0.2, random_state=42)
        
        self.capacity_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.capacity_model.fit(X_train, y_train)
        
        capacity_pred = self.capacity_model.predict(X_test)
        capacity_mae = mean_absolute_error(y_test, capacity_pred)
        capacity_r2 = r2_score(y_test, capacity_pred)
        
        # 효율성 예측 모델
        y_efficiency = data['efficiency']
        X_train, X_test, y_train, y_test = train_test_split(X, y_efficiency, test_size=0.2, random_state=42)
        
        self.efficiency_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.efficiency_model.fit(X_train, y_train)
        
        efficiency_pred = self.efficiency_model.predict(X_test)
        efficiency_mae = mean_absolute_error(y_test, efficiency_pred)
        efficiency_r2 = r2_score(y_test, efficiency_pred)
        
        # 전압 강하 예측 모델
        y_voltage = data['voltage_drop']
        X_train, X_test, y_train, y_test = train_test_split(X, y_voltage, test_size=0.2, random_state=42)
        
        self.voltage_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.voltage_model.fit(X_train, y_train)
        
        voltage_pred = self.voltage_model.predict(X_test)
        voltage_mae = mean_absolute_error(y_test, voltage_pred)
        voltage_r2 = r2_score(y_test, voltage_pred)
        
        return {
            'capacity': {'mae': capacity_mae, 'r2': capacity_r2},
            'efficiency': {'mae': efficiency_mae, 'r2': efficiency_r2},
            'voltage': {'mae': voltage_mae, 'r2': voltage_r2}
        }
    
    def predict_performance(self, cycles, temperature, c_rate, voltage, age_months):
        """
        주어진 조건에서 배터리 성능 예측
        """
        if not all([self.capacity_model, self.efficiency_model, self.voltage_model]):
            raise ValueError("모델이 학습되지 않았습니다. train_models()를 먼저 실행하세요.")
        
        input_data = np.array([[cycles, temperature, c_rate, voltage, age_months]])
        
        capacity = self.capacity_model.predict(input_data)[0]
        efficiency = self.efficiency_model.predict(input_data)[0]
        voltage_drop = self.voltage_model.predict(input_data)[0]
        
        return {
            'predicted_capacity': capacity,
            'predicted_efficiency': efficiency,
            'predicted_voltage': voltage_drop,
            'health_score': (capacity + efficiency) / 2 * 100
        }
    
    def visualize_predictions(self, data):
        """
        예측 결과 시각화
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('배터리 성능 예측 시스템 분석', fontsize=16, fontweight='bold')
        
        # 1. 사이클 vs 용량
        cycles_range = np.linspace(1, 2000, 100)
        capacity_predictions = []
        
        for cycle in cycles_range:
            pred = self.predict_performance(cycle, 25, 1.0, 3.7, 12)
            capacity_predictions.append(pred['predicted_capacity'])
        
        axes[0, 0].plot(cycles_range, capacity_predictions, 'b-', linewidth=2, label='예측값')
        axes[0, 0].scatter(data['cycles'][:100], data['capacity'][:100], alpha=0.5, c='red', s=20, label='실제값')
        axes[0, 0].set_xlabel('충방전 사이클')
        axes[0, 0].set_ylabel('용량 비율')
        axes[0, 0].set_title('사이클에 따른 용량 감소')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 온도 vs 효율성
        temp_range = np.linspace(-10, 60, 100)
        efficiency_predictions = []
        
        for temp in temp_range:
            pred = self.predict_performance(500, temp, 1.0, 3.7, 12)
            efficiency_predictions.append(pred['predicted_efficiency'])
        
        axes[0, 1].plot(temp_range, efficiency_predictions, 'g-', linewidth=2, label='예측값')
        axes[0, 1].scatter(data['temperature'][:100], data['efficiency'][:100], alpha=0.5, c='orange', s=20, label='실제값')
        axes[0, 1].set_xlabel('온도 (°C)')
        axes[0, 1].set_ylabel('효율성')
        axes[0, 1].set_title('온도에 따른 효율성 변화')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. C-rate vs 전압강하
        crate_range = np.linspace(0.1, 3.0, 100)
        voltage_predictions = []
        
        for crate in crate_range:
            pred = self.predict_performance(500, 25, crate, 4.0, 12)
            voltage_predictions.append(pred['predicted_voltage'])
        
        axes[0, 2].plot(crate_range, voltage_predictions, 'm-', linewidth=2, label='예측값')
        axes[0, 2].scatter(data['c_rate'][:100], data['voltage_drop'][:100], alpha=0.5, c='cyan', s=20, label='실제값')
        axes[0, 2].set_xlabel('C-rate')
        axes[0, 2].set_ylabel('전압 (V)')
        axes[0, 2].set_title('C-rate에 따른 전압 변화')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 특성 중요도 (용량)
        if hasattr(self.capacity_model, 'feature_importances_'):
            features = ['충방전\n사이클', '온도', 'C-rate', '전압', '사용기간']
            importances = self.capacity_model.feature_importances_
            
            bars = axes[1, 0].bar(features, importances, color='skyblue', alpha=0.7)
            axes[1, 0].set_ylabel('중요도')
            axes[1, 0].set_title('용량 예측 - 특성 중요도')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 막대 위에 값 표시
            for bar, importance in zip(bars, importances):
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{importance:.3f}', ha='center', va='bottom')
        
        # 5. 배터리 수명 시뮬레이션
        months = np.arange(0, 61, 1)
        health_scores = []
        
        for month in months:
            # 월별로 사이클 증가 가정 (월 30사이클)
            cycle = month * 30
            pred = self.predict_performance(cycle, 25, 1.0, 3.7, month)
            health_scores.append(pred['health_score'])
        
        axes[1, 1].plot(months, health_scores, 'r-', linewidth=3)
        axes[1, 1].axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='교체 권장선 (80%)')
        axes[1, 1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='교체 필수선 (70%)')
        axes[1, 1].set_xlabel('사용 기간 (개월)')
        axes[1, 1].set_ylabel('배터리 건강도 (%)')
        axes[1, 1].set_title('시간에 따른 배터리 건강도 변화')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(60, 100)
        
        # 6. 3D 성능 맵 (온도 vs C-rate vs 효율성)
        from mpl_toolkits.mplot3d import Axes3D
        
        axes[1, 2].remove()
        ax_3d = fig.add_subplot(2, 3, 6, projection='3d')
        
        temp_3d = np.linspace(0, 50, 20)
        crate_3d = np.linspace(0.5, 2.5, 20)
        temp_mesh, crate_mesh = np.meshgrid(temp_3d, crate_3d)
        
        efficiency_mesh = np.zeros_like(temp_mesh)
        for i in range(len(temp_3d)):
            for j in range(len(crate_3d)):
                pred = self.predict_performance(1000, temp_3d[i], crate_3d[j], 3.7, 24)
                efficiency_mesh[j, i] = pred['predicted_efficiency']
        
        surface = ax_3d.plot_surface(temp_mesh, crate_mesh, efficiency_mesh, 
                                   cmap='viridis', alpha=0.8)
        ax_3d.set_xlabel('온도 (°C)')
        ax_3d.set_ylabel('C-rate')
        ax_3d.set_zlabel('효율성')
        ax_3d.set_title('온도-C-rate 효율성 맵')
        
        plt.tight_layout()
        plt.show()

# 시스템 실행 및 데모
def main():
    print("🔋 배터리 성능 예측 시스템")
    print("=" * 50)
    
    # 예측 시스템 초기화
    predictor = BatteryPerformancePredictor()
    
    # 데이터 생성 및 모델 학습
    print("📊 합성 데이터 생성 중...")
    data = predictor.generate_synthetic_data(1000)
    
    print("🤖 AI 모델 학습 중...")
    metrics = predictor.train_models(data)
    
    print("\n📈 모델 성능:")
    for model_name, metric in metrics.items():
        print(f"{model_name:>10}: MAE={metric['mae']:.4f}, R²={metric['r2']:.4f}")
    
    # 예측 예시
    print("\n🔮 성능 예측 예시:")
    test_cases = [
        (100, 25, 1.0, 3.7, 6),    # 새 배터리
        (1000, 25, 1.0, 3.7, 24),  # 중간 수명
        (1800, 40, 2.0, 3.5, 48),  # 노후 배터리
    ]
    
    for i, (cycles, temp, c_rate, voltage, age) in enumerate(test_cases, 1):
        result = predictor.predict_performance(cycles, temp, c_rate, voltage, age)
        print(f"\n테스트 케이스 {i}:")
        print(f"  조건: {cycles}사이클, {temp}°C, C-rate {c_rate}, {voltage}V, {age}개월")
        print(f"  예측 용량: {result['predicted_capacity']:.1%}")
        print(f"  예측 효율성: {result['predicted_efficiency']:.1%}")
        print(f"  예측 전압: {result['predicted_voltage']:.2f}V")
        print(f"  건강도: {result['health_score']:.1f}%")
    
    # 시각화
    print("\n📊 결과 시각화 생성 중...")
    predictor.visualize_predictions(data)
    
    print("\n✅ 분석 완료! 그래프를 확인하세요.")

if __name__ == "__main__":
    main()