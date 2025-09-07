import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

print("=== SVM 5-Fold Cross-Validation Performance Evaluation ===")

# --- 1. 데이터 로드 및 전체 데이터 전처리 ---
try:
    # 이 스크립트와 같은 폴더에 features_left.csv, labels_left.csv 파일이 있어야 합니다.
    X = pd.read_csv('features_left.csv')
    y = pd.read_csv('labels_left.csv')
    y_flat = np.ravel(y)
except FileNotFoundError:
    print("오류: 'features_left.csv' 또는 'labels_left.csv' 파일을 찾을 수 없습니다.")
    exit()

# 라벨이 1, 2로 되어 있다면 0, 1로 변환
if set(np.unique(y_flat)) == {1, 2}:
    y_flat = y_flat - 1
    label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}
else:
    label_mapping = {0: 'Han (Cold)', 1: 'Yeol (Heat)'}

# 교차 검증을 위해 전체 데이터를 스케일링합니다.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n전체 데이터셋 정보:")
print(f"특성 개수: {X.shape[1]}")
print(f"총 샘플 수: {X.shape[0]}")

# --- 2. 모델 정의 ---
# GridSearchCV로 찾았던 최적의 파라미터를 사용
best_params = {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
model = SVC(**best_params, random_state=42)

# --- 3. 5-Fold 교차 검증 실행 ---
print("\n--- 5-Fold 교차 검증 실행 중 ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
cv_results = cross_validate(model, X_scaled, y_flat, cv=cv, scoring=scoring)

# --- 4. 교차 검증 결과 출력 ---
print("\n--- 5-Fold 교차 검증 평균 성능 ---")
print(f"평균 정확도 (Accuracy): {np.mean(cv_results['test_accuracy'])*100:.2f}% (±{np.std(cv_results['test_accuracy'])*100:.2f}%)")
print(f"평균 F1-Score:      {np.mean(cv_results['test_f1_macro']):.4f} (±{np.std(cv_results['test_f1_macro']):.4f})")
print(f"평균 정밀도 (Precision): {np.mean(cv_results['test_precision_macro']):.4f} (±{np.std(cv_results['test_precision_macro']):.4f})")
print(f"평균 재현율 (Recall):    {np.mean(cv_results['test_recall_macro']):.4f} (±{np.std(cv_results['test_recall_macro']):.4f})")

# --- 5. 전체 데이터에 대한 예측 및 종합 리포트 ---
print("\n--- 교차 검증 기반 종합 분류 리포트 ---")
y_pred_total = cross_val_predict(model, X_scaled, y_flat, cv=cv)
print(classification_report(y_flat, y_pred_total, target_names=list(label_mapping.values())))

# --- 6. 종합 혼동 행렬 시각화 ---
print("\n--- 종합 혼동 행렬 시각화 ---")
cm = confusion_matrix(y_flat, y_pred_total)
class_names = list(label_mapping.values())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Cross-Validated Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.savefig('SVM_CrossValidated_ConfusionMatrix.png', dpi=300, bbox_inches='tight')
plt.show()