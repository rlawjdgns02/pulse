import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 안전한 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=== Random Forest Hyperparameter Grid Search ===")
print("목표: 한/열 체질 분류 성능 최적화")

# ===== 설정 옵션 =====
# 탐색 범위 설정
SEARCH_MODE = 'full'  # 'fast', 'medium', 'full' 중 선택

# CPU 사용량 설정 (중요!)
CPU_USAGE = 'medium'     # 'low', 'medium', 'high' 중 선택

print(f"\n설정: {SEARCH_MODE} 탐색, {CPU_USAGE} CPU 사용")

# 1. 데이터 로드 및 전처리
print("\n1단계: 데이터 로드 및 전처리")
X = pd.read_csv('features_left.csv')
y = pd.read_csv('labels_left.csv')

print(f"데이터 크기: X{X.shape}, y{y.shape}")
print(f"클래스 분포:\n{y.value_counts()}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# y를 1차원으로 변환
y_train_flat = np.ravel(y_train)
y_test_flat = np.ravel(y_test)

print(f"훈련/테스트 분할: {X_train.shape[0]}/{X_test.shape[0]}")

# 2. 하이퍼파라미터 그리드 정의
print("\n2단계: 하이퍼파라미터 그리드 정의")

# 옵션 1: 빠른 탐색 (216개 조합, 약 2분)
param_grid_fast = {
    'n_estimators': [100, 200],                    # 2개 (기본적으로 좋은 값들)
    'max_depth': [3, 5, 7],                        # 3개 (과적합 방지)
    'min_samples_split': [2, 5],                   # 2개
    'min_samples_leaf': [1, 2],                    # 2개  
    'max_features': ['sqrt', 'log2'],              # 2개
    'class_weight': ['balanced', None]             # 2개 (가장 효과적)
}

# 옵션 2: 중간 탐색 (540개 조합, 약 5분)  
param_grid_medium = {
    'n_estimators': [50, 100, 200],                # 3개
    'max_depth': [3, 5, 7, None],                  # 4개
    'min_samples_split': [2, 5],                   # 2개
    'min_samples_leaf': [1, 2],                    # 2개
    'max_features': ['sqrt', 'log2'],              # 2개
    'class_weight': ['balanced', None]             # 2개
}

# 옵션 3: 전체 탐색 (1620개 조합, 약 15분)
param_grid_full = {
    'n_estimators': [50, 100, 200, 300],           # 4개
    'max_depth': [3, 5, 7, 10, None],              # 5개
    'min_samples_split': [2, 5, 10],               # 3개
    'min_samples_leaf': [1, 2, 4],                 # 3개
    'max_features': ['sqrt', 'log2', None],        # 3개
    'class_weight': ['balanced', 'balanced_subsample', None]  # 3개
}

# 원하는 옵션 선택
if SEARCH_MODE == 'fast':
    param_grid = param_grid_fast
    print("빠른 탐색 모드 선택 (216개 조합)")
elif SEARCH_MODE == 'medium':
    param_grid = param_grid_medium  
    print("중간 탐색 모드 선택 (540개 조합)")
else:
    param_grid = param_grid_full
    print("전체 탐색 모드 선택 (1620개 조합)")

# 총 조합 개수 계산
total_combinations = 1
for key, values in param_grid.items():
    total_combinations *= len(values)
    print(f"{key}: {values}")

print(f"\n총 하이퍼파라미터 조합 개수: {total_combinations}개")
print(f"5-fold CV 적용시 총 모델 학습 횟수: {total_combinations * 5}개")

# 3. CPU 사용량 제어 설정
print("\n3단계: CPU 사용량 설정")

if CPU_USAGE == 'low':
    # CPU 사용량 최소화 (1-2코어만 사용, 가장 느림)
    rf_n_jobs = 1
    grid_n_jobs = 1
    time_multiplier = 3.0
    print("저사용 모드: CPU 1-2코어만 사용 (다른 작업 가능)")
elif CPU_USAGE == 'medium':  
    # CPU 사용량 중간 (절반 코어 사용)
    cpu_count = os.cpu_count() or 4
    rf_n_jobs = 1
    grid_n_jobs = max(1, cpu_count // 2)  # 코어의 절반만 사용
    time_multiplier = 1.5
    print(f"중간 사용 모드: CPU {grid_n_jobs}코어 사용 (전체 {cpu_count}코어 중)")
else:
    # CPU 사용량 최대 (모든 코어 사용, 가장 빠름)
    rf_n_jobs = 1
    grid_n_jobs = -1
    time_multiplier = 1.0
    print("고사용 모드: 모든 CPU 코어 사용 (다른 작업 어려움)")

# 4. GridSearchCV 설정
print("\n4단계: GridSearchCV 설정")

# 다중 평가 지표 설정
scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro'
}

# 5-fold 층화 교차검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV 객체 생성 (F1-score를 주 평가지표로 사용)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=rf_n_jobs),
    param_grid=param_grid,
    cv=cv,
    scoring=scoring,
    refit='f1_macro',  # 최적 모델 선택 기준
    n_jobs=grid_n_jobs,  # 제한된 병렬처리
    verbose=1,           # 진행상황 출력
    return_train_score=True
)

# 예상 시간 계산
estimated_minutes = (total_combinations * 5 * 0.1 / 60) * time_multiplier

print("GridSearchCV 설정 완료:")
print(f"- 교차검증: {cv.n_splits}-fold Stratified")
print(f"- 평가지표: {list(scoring.keys())}")
print(f"- 재학습 기준: f1_macro")
print(f"- CPU 사용량: {CPU_USAGE} 모드 ({grid_n_jobs if grid_n_jobs != -1 else 'all'} 코어)")
print(f"- 예상 소요시간: 약 {estimated_minutes:.1f}분")

# 5. Grid Search 실행
print("\n5단계: Grid Search 실행")
print("⏰ 진행상황을 확인하세요... (Ctrl+C로 중단 가능)")

start_time = time.time()

try:
    grid_search.fit(X_train_scaled, y_train_flat)
    search_time = time.time() - start_time
    
    print(f"\n✅ Grid Search 완료!")
    print(f"- 실제 소요 시간: {search_time/60:.1f}분")
    print(f"- 테스트된 조합: {len(grid_search.cv_results_['params'])}개")
    
except KeyboardInterrupt:
    print(f"\n⛔ 사용자가 중단했습니다.")
    print(f"- 진행 시간: {(time.time() - start_time)/60:.1f}분")
    exit()

# 6. 최적 결과 출력
print("\n=== 6단계: 최적 하이퍼파라미터 결과 ===")

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("🎯 최적 하이퍼파라미터:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n📊 최고 교차검증 F1-Score: {best_score:.4f}")

# 최적 모델로 테스트 데이터 평가
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)

test_accuracy = accuracy_score(y_test_flat, y_pred_best)
test_f1 = f1_score(y_test_flat, y_pred_best, average='macro')

print(f"\n🧪 테스트 데이터 성능:")
print(f"  정확도: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"  F1-Score: {test_f1:.4f}")

print(f"\n📋 상세 분류 보고서:")
print(classification_report(y_test_flat, y_pred_best))

# 7. 상위 모델들 비교
print("\n=== 7단계: 상위 성능 모델들 비교 ===")

results_df = pd.DataFrame(grid_search.cv_results_)

# F1-score 기준 상위 10개 모델
top_models = results_df.nlargest(10, 'mean_test_f1_macro')

print("🏆 F1-Score 기준 상위 10개 모델:")
print("-" * 120)
print(f"{'순위':<4} {'F1-Score':<10} {'정확도':<8} {'n_est':<6} {'depth':<6} {'split':<6} {'leaf':<5} {'features':<8} {'weight':<12}")
print("-" * 120)

for idx, (i, row) in enumerate(top_models.iterrows()):
    print(f"{idx+1:<4} {row['mean_test_f1_macro']:<10.4f} "
          f"{row['mean_test_accuracy']:<8.4f} "
          f"{row['param_n_estimators']:<6} "
          f"{str(row['param_max_depth']):<6} "
          f"{row['param_min_samples_split']:<6} "
          f"{row['param_min_samples_leaf']:<5} "
          f"{str(row['param_max_features']):<8} "
          f"{str(row['param_class_weight']):<12}")

# 8. 특성 중요도 분석
print("\n=== 8단계: 최적 모델의 특성 중요도 ===")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("🔍 상위 10개 중요한 특성:")
print("-" * 50)
for i in range(min(10, len(feature_importance))):
    row = feature_importance.iloc[i]
    print(f"{i+1:2d}. {row['feature']:<30} : {row['importance']:.4f}")

# 9. 결과 시각화 함수
def plot_grid_search_results(grid_search, param1, param2, score='f1_macro'):
    """그리드 서치 결과를 히트맵으로 시각화"""
    
    results_df = pd.DataFrame(grid_search.cv_results_)
    score_key = f'mean_test_{score}'
    
    # 컬럼명 확인
    param1_col = f'param_{param1}'
    param2_col = f'param_{param2}'
    
    if param1_col in results_df.columns and param2_col in results_df.columns:
        # 유니크한 값들이 충분한지 확인
        unique_param1 = results_df[param1_col].nunique()
        unique_param2 = results_df[param2_col].nunique()
        
        if unique_param1 > 1 and unique_param2 > 1:
            pivot_table = results_df.pivot_table(
                values=score_key,
                index=param1_col,
                columns=param2_col,
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis',
                        cbar_kws={'label': f'{score.replace("_", " ").title()}'})
            
            plt.title(f'Random Forest Grid Search Results\n({param1} vs {param2})', 
                      fontsize=14, fontweight='bold')
            plt.xlabel(param2.replace('_', ' ').title(), fontweight='bold')
            plt.ylabel(param1.replace('_', ' ').title(), fontweight='bold')
            
            plt.tight_layout()
            filename = f'rf_gridsearch_{param1}_{param2}_{score}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"📈 히트맵이 '{filename}'으로 저장되었습니다.")
            plt.show()
            
            return pivot_table
        else:
            print(f"⚠️ 파라미터 조합이 충분하지 않아 히트맵을 생성할 수 없습니다.")
            return None
    else:
        print(f"⚠️ 파라미터 '{param1}' 또는 '{param2}'를 찾을 수 없습니다.")
        return None

# 10. 주요 파라미터 조합 시각화
print("\n=== 9단계: 결과 시각화 ===")

try:
    # n_estimators vs max_depth 히트맵
    plot_grid_search_results(grid_search, 'n_estimators', 'max_depth', 'f1_macro')
    
    # min_samples_split vs min_samples_leaf 히트맵  
    plot_grid_search_results(grid_search, 'min_samples_split', 'min_samples_leaf', 'f1_macro')
except Exception as e:
    print(f"⚠️ 시각화 중 오류 발생: {e}")
    print("히트맵 생성을 건너뜁니다.")

# 11. 성능 향상 요약
print("\n=== 🎉 최종 요약 ===")
print(f"기존 Random Forest 성능:     67% (F1: 0.66)")
print(f"최적화된 Random Forest 성능:  {test_accuracy*100:.1f}% (F1: {test_f1:.4f})")

improvement_acc = test_accuracy*100 - 67
improvement_f1 = test_f1 - 0.66

if improvement_acc > 0:
    print(f"🚀 성능 향상:                   +{improvement_acc:.1f}%p (F1: {improvement_f1:+.4f})")
else:
    print(f"📉 성능 변화:                   {improvement_acc:.1f}%p (F1: {improvement_f1:+.4f})")

print(f"\n💡 추천 하이퍼파라미터:")
for param, value in best_params.items():
    print(f"  {param}: {value}")

print(f"\n⏱️ 총 소요 시간: {search_time/60:.1f}분")
print(f"💻 CPU 사용 모드: {CPU_USAGE}")
print(f"🔍 탐색 범위: {SEARCH_MODE}")

print("\n📝 참고사항:")
print("- 소규모 데이터셋(90개)에서는 과적합 주의가 필요합니다")
print("- 더 많은 데이터 수집으로 성능 향상이 가능합니다")
print("- 다른 특성 조합이나 전처리 방법도 고려해보세요")

# 모든 플롯 창 닫기
plt.close('all')