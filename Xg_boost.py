from xgboost import XGBClassifier, plot_importance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

# XGBClassifier 파라미터 설정 (작은 데이터셋용)
model = XGBClassifier(
    booster='gbtree',
    verbosity=0,  # silent 대신 verbosity 사용
    n_jobs=4,     # nthread 대신 n_jobs 사용
    random_state=42,
    eval_metric='logloss',
    # 작은 데이터셋에 적합한 파라미터
    max_depth=3,          # 과적합 방지를 위해 깊이 제한
    n_estimators=100,     # 트리 개수 줄임
    learning_rate=0.1,    # 학습률
    subsample=0.8,        # 샘플링 비율
    colsample_bytree=0.8  # 특성 샘플링 비율
)

# 모델 훈련 (작은 데이터셋이므로 early stopping 제거)
print("Training XGBoost model...")
xgb_model = model.fit(X_train, y_train)

# 예측 및 평가
y_pred = xgb_model.predict(X_test)
y_pred_probs = xgb_model.predict_proba(X_test)[:, 1]

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Model Performance ===")
print(f"Test Accuracy: {accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=list(label_mapping.values())))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Cross-validation으로 더 robust한 평가
print(f"\n=== Cross-Validation Results ===")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual CV scores: {cv_scores}")

# 특성 중요도 시각화
print("\n=== Feature Importance Visualization ===")
fig, ax = plt.subplots(figsize=(10, 12))
plot_importance(xgb_model, ax=ax, max_num_features=20)  # 상위 20개 특성만 표시
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()

# 특성 중요도를 DataFrame으로 출력
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Most Important Features ===")
print(feature_importance.head(10))

# 모델 훈련 완료 메시지
print(f"\nTraining completed with {model.n_estimators} estimators")