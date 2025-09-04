import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import time

print("=== Random Forest 한/열 체질 분류 ===")
print("1단계: 라이브러리 로드 완료")

# 데이터 로드 (우수 데이터셋 사용 - SVM에서 75% 나왔던 데이터)
X = pd.read_csv('features_left.csv')  # 좌수 데이터
y = pd.read_csv('labels_left.csv')

print(f"\n데이터 로드 완료:")
print(f"- X 데이터 크기: {X.shape}")
print(f"- y 데이터 크기: {y.shape}")
print(f"- 특성 수: {X.shape[1]}개")
print(f"- 샘플 수: {X.shape[0]}개")

# 클래스 분포 확인
print(f"\n클래스 분포:")
print(y.value_counts())

# ===== 2단계: 데이터 전처리 =====
print("\n=== 2단계: 데이터 전처리 시작 ===")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # 클래스 비율 유지
)

print(f"데이터 분할 완료:")
print(f"- 훈련 데이터: {X_train.shape}")
print(f"- 테스트 데이터: {X_test.shape}")

# MinMaxScaler로 모든 특성을 0~1 범위로 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n스케일링 완료:")
print(f"- 모든 특성이 0~1 범위로 변환됨")
print(f"- 스케일링 전 예시: 첫 번째 특성 범위 [{X_train.iloc[:, 0].min():.2f}, {X_train.iloc[:, 0].max():.2f}]")
print(f"- 스케일링 후 예시: 첫 번째 특성 범위 [{X_train_scaled[:, 0].min():.2f}, {X_train_scaled[:, 0].max():.2f}]")

# y 데이터를 1차원으로 변환 (sklearn 요구사항)
y_train_flat = np.ravel(y_train)
y_test_flat = np.ravel(y_test)

print(f"\ny 데이터 변환 완료:")
print(f"- 훈련 y: {y_train_flat.shape}")
print(f"- 테스트 y: {y_test_flat.shape}")

# ===== 3단계: Random Forest 모델 정의 =====
print("\n=== 3단계: Random Forest 모델 정의 ===")

# 작은 데이터셋에 최적화된 Random Forest 설정
rf_model = RandomForestClassifier(
    n_estimators=300,        # 트리 개수: 100개 (적당함)
    max_depth=3,             # 트리 깊이: 5 (과적합 방지)
    min_samples_split=10,     # 분할 최소 샘플: 5개 (과적합 방지)
    min_samples_leaf=1,      # 리프 노드 최소 샘플: 2개
    # max_features='sqrt',     # 사용 특성 수: √19 ≈ 4개
    random_state=42,         # 재현성을 위한 시드
    class_weight='balanced_subsample', # 클래스 불균형 대응
    n_jobs=-1               # 병렬 처리 (속도 향상)
)

print("Random Forest 모델 설정:")
print(f"- 트리 개수: {rf_model.n_estimators}개")
print(f"- 최대 깊이: {rf_model.max_depth}")
print(f"- 분할 최소 샘플: {rf_model.min_samples_split}")
print(f"- 리프 최소 샘플: {rf_model.min_samples_leaf}")
print(f"- 사용 특성 수: {rf_model.max_features} (√{X.shape[1]} ≈ {int(np.sqrt(X.shape[1]))}개)")
print(f"- 클래스 가중치: {rf_model.class_weight}")

# ===== 4단계: 모델 학습 =====
print("\n=== 4단계: 모델 학습 ===")

# 학습 시간 측정
start_time = time.time()

# Random Forest 학습
rf_model.fit(X_train_scaled, y_train_flat)

training_time = time.time() - start_time

print(f"학습 완료!")
print(f"- 학습 시간: {training_time:.4f}초")
print(f"- 훈련된 트리 개수: {len(rf_model.estimators_)}")

# ===== 5단계: 예측 및 평가 =====
print("\n=== 5단계: 예측 및 평가 ===")

# 테스트 데이터로 예측
y_pred = rf_model.predict(X_test_scaled)

# 성능 평가
print("--- Random Forest 성능 평가 결과 ---")
print(classification_report(y_test_flat, y_pred))

print("\n[오차 행렬 (Confusion Matrix)]")
print(confusion_matrix(y_test_flat, y_pred))

# SVM과 비교
print("\n--- SVM vs Random Forest 비교 ---")
from sklearn.metrics import accuracy_score, f1_score
rf_accuracy = accuracy_score(y_test_flat, y_pred)
rf_f1 = f1_score(y_test_flat, y_pred, average='macro')

print(f"SVM 성능 (우수 19개 특성):     정확도 75%")
print(f"Random Forest 성능:          정확도 {rf_accuracy*100:.0f}%")
print(f"성능 차이:                   {rf_accuracy*100 - 75:+.0f}%p")

# ===== 6단계: 특성 중요도 분석 =====
print("\n=== 6단계: 특성 중요도 분석 ===")

# 특성 중요도 추출
importances = rf_model.feature_importances_
feature_names = X.columns

# 중요도와 특성명을 함께 정렬
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n상위 10개 중요한 특성:")
print("=" * 40)
for i in range(min(10, len(feature_importance_df))):
    row = feature_importance_df.iloc[i]
    print(f"{i+1:2d}. {row['feature']:<25} : {row['importance']:.4f}")

print(f"\n전체 특성 중요도 합계: {importances.sum():.4f} (1.0이어야 정상)")

# ===== 7단계: 교차검증 =====
print("\n=== 7단계: 교차검증으로 안정성 확인 ===")

# 5-Fold 층화 교차검증
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# F1-score로 교차검증
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train_flat, cv=cv, scoring='f1_macro')

print("5-Fold 교차검증 결과 (F1-macro):")
print(f"- 각 폴드 성능: {cv_scores}")
print(f"- 평균 성능:    {cv_scores.mean():.4f}")
print(f"- 표준편차:     {cv_scores.std():.4f}")
print(f"- 95% 신뢰구간: {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")

# 정확도로도 교차검증
cv_accuracy = cross_val_score(rf_model, X_train_scaled, y_train_flat, cv=cv, scoring='accuracy')
print(f"\n정확도 교차검증:")
print(f"- 평균 정확도:  {cv_accuracy.mean()*100:.1f}%")
print(f"- 표준편차:     {cv_accuracy.std()*100:.1f}%")