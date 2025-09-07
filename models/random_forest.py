import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=== RandomForest 5-Fold Cross-Validation Performance Evaluation ===")

# --- 1. 데이터 로드 및 전체 데이터 전처리 ---
try:
    X = pd.read_csv('features_left.csv')
    y = pd.read_csv('labels_left.csv')
    y_flat = np.ravel(y)
except FileNotFoundError:
    print("Error: 'features_left.csv' or 'labels_left.csv' not found.")
    exit()

# 라벨 1, 2 -> 0, 1 변환
y_flat = y_flat - 1 if set(np.unique(y_flat)) == {1, 2} else y_flat
label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset Info: Features={X.shape[1]}, Samples={X.shape[0]}")

# --- 2. 최적 파라미터로 모델 정의 ---
# 이전 GridSearchCV에서 찾은 최적의 파라미터 사용
best_params = {
    'class_weight': 'balanced',
    'max_depth': 3,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'min_samples_split': 10,
    'n_estimators': 50
}
model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)

# --- 3. 5-Fold 교차 검증 실행 ---
print("\n--- Running 5-Fold Cross-Validation ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
cv_results = cross_validate(model, X_scaled, y_flat, cv=cv, scoring=scoring)

# --- 4. 교차 검증 결과 출력 ---
print("\n--- 5-Fold Cross-Validation Average Performance ---")
print(f"Average Accuracy:  {np.mean(cv_results['test_accuracy'])*100:.2f}% (±{np.std(cv_results['test_accuracy'])*100:.2f}%)")
print(f"Average F1-Score:      {np.mean(cv_results['test_f1_macro']):.4f} (±{np.std(cv_results['test_f1_macro']):.4f})")

# --- 5. 종합 리포트 및 혼동 행렬 ---
print("\n--- Cross-Validated Classification Report ---")
y_pred_total = cross_val_predict(model, X_scaled, y_flat, cv=cv)
print(classification_report(y_flat, y_pred_total, target_names=list(label_mapping.values())))

cm = confusion_matrix(y_flat, y_pred_total)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.title('RandomForest Cross-Validated Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
