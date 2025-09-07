import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print("=== AdaBoost 5-Fold Cross-Validation & Hyperparameter Tuning ===")

# --- 1. 데이터 로드 및 전체 데이터 전처리 ---
try:
    X = pd.read_csv('data/all_features_left.csv')
    y = pd.read_csv('data/labels_all_left.csv')
    y_flat = np.ravel(y)
except FileNotFoundError:
    print("Error: 'features_left.csv' or 'labels_left.csv' not found.")
    exit()

y_flat = y_flat - 1 if set(np.unique(y_flat)) == {1, 2} else y_flat
label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nDataset Info: Features={X.shape[1]}, Samples={X.shape[0]}")

# --- 2. 하이퍼파라미터 그리드 정의 및 탐색 (GridSearchCV) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.5, 1.0]
}

print("\n--- Running GridSearchCV to find best parameters ---")
grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(random_state=42),
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("\n--- Best Parameters Found ---")
print(best_params)

# --- 3. 찾은 최적 파라미터로 모델 최종 정의 ---
model = AdaBoostClassifier(**best_params, random_state=42)

# --- 4. 전체 데이터셋으로 5-Fold 교차 검증 실행 ---
print("\n--- Running 5-Fold Cross-Validation with Best Parameters ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
cv_results = cross_validate(model, X_scaled, y_flat, cv=cv, scoring=scoring)

# --- 5. 교차 검증 결과 출력 ---
print("\n--- 5-Fold Cross-Validation Average Performance ---")
print(f"Average Accuracy:  {np.mean(cv_results['test_accuracy'])*100:.2f}% (±{np.std(cv_results['test_accuracy'])*100:.2f}%)")
print(f"Average F1-Score:      {np.mean(cv_results['test_f1_macro']):.4f} (±{np.std(cv_results['test_f1_macro']):.4f})")

# --- 6. 종합 리포트 및 혼동 행렬 ---
print("\n--- Cross-Validated Classification Report ---")
y_pred_total = cross_val_predict(model, X_scaled, y_flat, cv=cv)
print(classification_report(y_flat, y_pred_total, target_names=list(label_mapping.values())))

cm = confusion_matrix(y_flat, y_pred_total)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
plt.title('AdaBoost Cross-Validated Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

