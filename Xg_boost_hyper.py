from xgboost import XGBClassifier, plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import time
import os
warnings.filterwarnings('ignore')

print("=== XGBoost CPU 사용량 제어 GridSearch ===")

# ===== 🔧 CPU 사용량 설정 (중요!) =====
CPU_USAGE = 'low'     # 'low', 'medium', 'high' 중 선택
SEARCH_MODE = 'full'  # 'fast', 'medium', 'full' 중 선택

print(f"설정: {SEARCH_MODE} 탐색, {CPU_USAGE} CPU 사용")

# 데이터 로드
X = pd.read_csv('features_left.csv')
y = pd.read_csv('labels_left.csv')
y = y.astype(int)

# 라벨 값 확인 및 안전한 처리
print(f"Original label distribution: {np.bincount(np.ravel(y))}")
print(f"Unique labels: {np.unique(np.ravel(y))}")

# 라벨이 1, 2로 되어 있다면 0, 1로 변환 (scikit-learn 호환성)
y_flat = np.ravel(y)
if set(np.unique(y_flat)) == {1, 2}:
    print("Converting labels from {1, 2} to {0, 1} for sklearn compatibility")
    y_flat = y_flat - 1  # 1,2 → 0,1 변환
    label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}
    original_labels = {1: 'Han (Cold)', 2: 'Yeol (Heat)'}
else:
    label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}
    original_labels = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}

print(f"Final label distribution: {np.bincount(y_flat)}")

# 데이터 분할 - 작은 데이터셋이므로 train/test만 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Total dataset size: {len(y_flat)}")

# ===== 📊 하이퍼파라미터 그리드 정의 =====
print("\n하이퍼파라미터 그리드 정의...")

# 옵션 1: 빠른 탐색 (48개 조합, 약 2-5분)
param_grid_fast = {
    'max_depth': [3, 4, 5],                    # 3개
    'n_estimators': [50, 100],                 # 2개
    'learning_rate': [0.1, 0.15],              # 2개
    'subsample': [0.8, 1.0],                   # 2개
    'colsample_bytree': [0.8, 1.0],            # 2개
}

# 옵션 2: 중간 탐색 (192개 조합, 약 8-15분)
param_grid_medium = {
    'max_depth': [3, 4, 5, 6],                 # 4개
    'n_estimators': [50, 100, 150],            # 3개
    'learning_rate': [0.05, 0.1, 0.15],        # 3개
    'subsample': [0.7, 0.8, 1.0],              # 3개
    'colsample_bytree': [0.8, 1.0],            # 2개
    'min_child_weight': [1, 3]                 # 2개
}

# 옵션 3: 전체 탐색 (864개 조합, 약 30-60분)
param_grid_full = {
    'max_depth': [2, 3, 4, 5, 6],              # 5개
    'n_estimators': [50, 100, 150, 200],       # 4개
    'learning_rate': [0.05, 0.1, 0.15, 0.2],   # 4개
    'subsample': [0.7, 0.8, 0.9, 1.0],         # 4개
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],  # 4개
    'min_child_weight': [1, 3, 5],             # 3개
    'gamma': [0, 0.1, 0.2]                     # 3개
}

# 원하는 옵션 선택
if SEARCH_MODE == 'fast':
    param_grid = param_grid_fast
    print("빠른 탐색 모드 선택")
elif SEARCH_MODE == 'medium':
    param_grid = param_grid_medium  
    print("중간 탐색 모드 선택")
else:
    param_grid = param_grid_full
    print("전체 탐색 모드 선택")

# 총 조합 개수 계산
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)
    print(f"{key}: {values}")

print(f"\n총 하이퍼파라미터 조합 개수: {total_combinations}개")

# ===== 💻 CPU 사용량 제어 설정 =====
print("\nCPU 사용량 설정...")

if CPU_USAGE == 'low':
    # CPU 사용량 최소화 (1코어만 사용, 가장 느림, 다른 작업 가능)
    xgb_n_jobs = 1
    grid_n_jobs = 1
    time_multiplier = 3.0
    print("🐌 저사용 모드: CPU 1코어만 사용 (다른 작업 가능)")
    
elif CPU_USAGE == 'medium':  
    # CPU 사용량 중간 (절반 코어 사용)
    cpu_count = os.cpu_count() or 4
    xgb_n_jobs = 1
    grid_n_jobs = max(1, cpu_count // 2)  # 코어의 절반만 사용
    time_multiplier = 1.5
    print(f"⚖️ 중간 사용 모드: CPU {grid_n_jobs}코어 사용 (전체 {cpu_count}코어 중)")
    
else:
    # CPU 사용량 최대 (모든 코어 사용, 가장 빠름)
    xgb_n_jobs = 4
    grid_n_jobs = -1
    time_multiplier = 1.0
    print("🚀 고사용 모드: 모든 CPU 코어 사용 (다른 작업 어려움)")

# 예상 시간 계산
estimated_minutes = (total_combinations * 3 * 0.05 / 60) * time_multiplier
print(f"⏱️ 예상 소요시간: 약 {estimated_minutes:.1f}분")

# ===== 🎯 XGBClassifier 파라미터 설정 =====
base_model = XGBClassifier(
    booster='gbtree',
    verbosity=0,  # 로그 출력 최소화
    n_jobs=xgb_n_jobs,  # XGBoost 자체 CPU 사용량 제한
    random_state=42,
    eval_metric='logloss'
)

# ===== 🔍 GridSearchCV 설정 =====
print(f"\nGridSearchCV 설정 중...")
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,  # 작은 데이터셋이므로 3-fold
    n_jobs=grid_n_jobs,  # 제한된 병렬처리 ⭐️ 핵심!
    verbose=1,  # 진행상황 출력
    return_train_score=True
)

print("GridSearchCV 설정 완료:")
print(f"- 교차검증: 3-fold")
print(f"- CPU 사용량: {CPU_USAGE} 모드 ({grid_n_jobs if grid_n_jobs != -1 else 'all'} 코어)")
print(f"- XGBoost 자체: {xgb_n_jobs} 코어")

# ===== 🚀 Grid Search 실행 =====
print(f"\n=== GridSearch 시작 ===")
print(f"🔄 {total_combinations}개 조합 테스트 중...")
print("⏰ 진행상황을 확인하세요... (Ctrl+C로 중단 가능)")

start_time = time.time()

try:
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time
    
    print(f"\n✅ GridSearch 완료!")
    print(f"- 실제 소요 시간: {search_time/60:.1f}분")
    print(f"- 테스트된 조합: {len(grid_search.cv_results_['params'])}개")
    
except KeyboardInterrupt:
    print(f"\n⛔ 사용자가 중단했습니다.")
    print(f"- 진행 시간: {(time.time() - start_time)/60:.1f}분")
    exit()

# ===== 📊 최적 파라미터 결과 출력 =====
print("\n=== 최적 하이퍼파라미터 결과 ===")

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("🎯 최적 하이퍼파라미터:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 최고 교차검증 정확도: {best_score:.4f}")

# 최적 모델로 예측
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_pred_probs = best_model.predict_proba(X_test)[:, 1]

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🧪 테스트 데이터 성능:")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.values())))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# ===== 🏆 Top 10 파라미터 조합 분석 =====
print(f"\n=== Top 10 Parameter Combinations ===")
results_df = pd.DataFrame(grid_search.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]

print("🏆 상위 10개 모델:")
print("-" * 80)
for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
    params_str = ', '.join([f"{k.replace('param_', '')}: {v}" for k, v in row['params'].items()])
    print(f"    {params_str}")

# ===== 📈 특성 중요도 시각화 =====
print("\n=== Feature Importance Visualization (Best Model) ===")
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(best_model, ax=ax, max_num_features=15)
    ax.set_title('XGBoost Feature Importance (Best Model)')
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"시각화 오류: {e}")

# 특성 중요도를 DataFrame으로 출력
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Most Important Features (Best Model) ===")
for i in range(min(10, len(feature_importance))):
    row = feature_importance.iloc[i]
    print(f"{i+1:2d}. {row['feature']:<30} : {row['importance']:.4f}")

# ===== 🎉 최종 요약 =====
print(f"\n=== 🎉 최종 요약 ===")

# 성능 향상 비교 (기본 모델 vs 최적 모델)
print(f"성능 비교:")
default_model = XGBClassifier(random_state=42, verbosity=0, n_jobs=1)
default_model.fit(X_train, y_train)
default_pred = default_model.predict(X_test)
default_accuracy = accuracy_score(y_test, default_pred)

print(f"- 기본 XGBoost 정확도:     {default_accuracy:.4f} ({default_accuracy*100:.1f}%)")
print(f"- 최적화된 XGBoost 정확도:  {accuracy:.4f} ({accuracy*100:.1f}%)")

improvement = accuracy - default_accuracy
if improvement > 0:
    print(f"🚀 성능 향상: +{improvement:.4f} (+{improvement*100:.1f}%p)")
else:
    print(f"📉 성능 변화: {improvement:.4f} ({improvement*100:.1f}%p)")

print(f"\n💡 추천 하이퍼파라미터:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n⏱️ 총 소요 시간: {search_time/60:.1f}분")
print(f"💻 CPU 사용 모드: {CPU_USAGE}")
print(f"🔍 탐색 범위: {SEARCH_MODE}")

print(f"\n📝 CPU 사용량 제어 팁:")
print(f"- 'low': 백그라운드 실행용 (1코어, 느림)")
print(f"- 'medium': 일반 사용 (절반 코어, 적당)")  
print(f"- 'high': 최대 성능 (모든 코어, 빠름)")