import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# 영어 폰트 설정
plt.style.use('default')
plt.rcParams['font.size'] = 10

# 1. 데이터 로드 및 전처리
print("=== SVM 한/열 체질 분류 결과 시각화 ===\n")

X = pd.read_csv('features_right.csv')
y = pd.read_csv('labels_right.csv')
y = y.astype(int)

# ⭐️ 수정 1: 라벨 값 확인 및 안전한 처리
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

X_train, X_test, y_train, y_test = train_test_split(X, y_flat, test_size=0.2, random_state=42, stratify=y_flat)

# 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set distribution: {np.bincount(y_train)}")
print(f"Test set distribution: {np.bincount(y_test)}")

# 2. SVM 모델 학습
best_params = {'C': 1000, 'kernel': 'linear'}
best_svm_model = SVC(**best_params, random_state=42)
best_svm_model.fit(X_train_scaled, y_train)
y_pred = best_svm_model.predict(X_test_scaled)

# 3. 성능 지표 계산
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

print(f"\nDataset: Left-side Pulse Wave Data")
print(f"Features: {X.shape[1]}")
print(f"Total Samples: {X.shape[0]} (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
print(f"SVM Settings: C={best_params['C']}, kernel={best_params['kernel']}")

# 4. 시각화
fig = plt.figure(figsize=(16, 12))

# 4-1. Performance Metrics Bar Chart
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [accuracy, precision, recall, f1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax1.bar(metrics, [v*100 for v in values], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
ax1.set_ylabel('Performance (%)', fontsize=12)
ax1.set_title('SVM Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)

# 바 위에 수치 표시
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{value*100:.1f}%', ha='center', va='bottom', fontweight='bold')

ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 4-2. Confusion Matrix Heatmap
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred)

# ⭐️ 수정 2: 클래스 이름을 실제 라벨에 맞게 설정
class_names = list(label_mapping.values())

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=ax2)
ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted Class', fontsize=12)
ax2.set_ylabel('Actual Class', fontsize=12)

# 4-3. Normalized Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
# ⭐️ 수정 3: 안전한 정규화 (0으로 나누기 방지)
cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Reds',
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Accuracy Ratio'}, ax=ax3)
ax3.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Predicted Class', fontsize=12)
ax3.set_ylabel('Actual Class', fontsize=12)

# 4-4. Class-wise Performance Comparison
ax4 = plt.subplot(2, 3, 4)
class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

# ⭐️ 수정 4: 클래스 인덱스를 문자열로 안전하게 변환
class_keys = ['0', '1']  # sklearn은 클래스를 문자열로 저장
classes = [f"{label_mapping[0]}", f"{label_mapping[1]}"]

# 클래스별 성능 지표 안전하게 추출
class_precision = []
class_recall = []
class_f1 = []

for key in class_keys:
    if key in class_report:
        class_precision.append(class_report[key]['precision'])
        class_recall.append(class_report[key]['recall'])
        class_f1.append(class_report[key]['f1-score'])
    else:
        class_precision.append(0)
        class_recall.append(0)
        class_f1.append(0)

x = np.arange(len(classes))
width = 0.25

bars1 = ax4.bar(x - width, [p*100 for p in class_precision], width, label='Precision', color='#FF6B6B', alpha=0.8)
bars2 = ax4.bar(x, [r*100 for r in class_recall], width, label='Recall', color='#4ECDC4', alpha=0.8)
bars3 = ax4.bar(x + width, [f*100 for f in class_f1], width, label='F1-Score', color='#45B7D1', alpha=0.8)

ax4.set_ylabel('Performance (%)', fontsize=12)
ax4.set_title('Class-wise Performance Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(classes)
ax4.legend()
ax4.set_ylim(0, 100)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 수치 표시
for bars, values in zip([bars1, bars2, bars3], [class_precision, class_recall, class_f1]):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value*100:.1f}%', ha='center', va='bottom', fontsize=9)

# 4-5. Data Distribution Pie Chart
ax5 = plt.subplot(2, 3, 5)
class_counts = np.bincount(y_test)
# ⭐️ 수정 5: 클래스 카운트가 0인 경우 처리
if len(class_counts) >= 2:
    labels = [f'{label_mapping[0]}\n({class_counts[0]} samples)', 
              f'{label_mapping[1]}\n({class_counts[1] if len(class_counts) > 1 else 0} samples)']
    sizes = [class_counts[0], class_counts[1] if len(class_counts) > 1 else 0]
else:
    labels = [f'{label_mapping[0]}\n({class_counts[0]} samples)']
    sizes = [class_counts[0]]

colors = ['#FFE5E5', '#E5F3FF']
explode = (0.05, 0.05) if len(sizes) == 2 else (0.05,)

wedges, texts, autotexts = ax5.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                   colors=colors[:len(sizes)], explode=explode[:len(sizes)], 
                                   shadow=True, textprops={'fontsize': 11})
ax5.set_title('Test Data Class Distribution', fontsize=14, fontweight='bold')

# 4-6. Summary Results Text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
📊 SVM Han/Yeol Constitution Classification Summary

🔹 Dataset Information
   • Left-side pulse wave data
   • Total features: {X.shape[1]}
   • Total samples: {X.shape[0]}

🔹 Model Configuration
   • Algorithm: Support Vector Machine
   • Kernel: RBF (Radial Basis Function)
   • Regularization parameter C: {best_params['C']}

🔹 Performance Results
   • Overall accuracy: {accuracy*100:.1f}%
   • Macro F1-Score: {f1*100:.1f}%
   
🔹 Class-wise Performance
   • {label_mapping[0]} F1-score: {class_f1[0]*100:.1f}%
   • {label_mapping[1]} F1-score: {class_f1[1]*100:.1f}%

🔹 Model Characteristics
   • Classification performance analysis
   • Balanced train/test split maintained
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.92, bottom=0.08)
plt.suptitle('SVM Model Performance Analysis for Han/Yeol Constitution Classification', 
             fontsize=15, fontweight='bold')

# 5. Save and Display
plt.savefig('SVM_HanYeol_Classification_Results.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Detailed Results Output
print("\n" + "="*60)
print("📋 Detailed Classification Report")
print("="*60)
# ⭐️ 수정 6: target_names를 실제 라벨에 맞게 설정
target_names = [label_mapping[0].split(' ')[0], label_mapping[1].split(' ')[0]]
print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

print("\n" + "="*60)
print("🎯 Key Results Summary")
print("="*60)
print(f"• Overall accuracy: {accuracy*100:.1f}%")
print(f"• {target_names[0]} F1-score: {class_f1[0]*100:.1f}%")
print(f"• {target_names[1]} F1-score: {class_f1[1]*100:.1f}%")
print(f"• Average F1-score: {f1*100:.1f}%")

# ⭐️ 수정 7: Confusion Matrix 해석을 2x2 행렬에 맞게 수정
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\n📊 Confusion Matrix Interpretation:")
    print(f"• {target_names[0]} correct: {tn}, {target_names[0]} misclassified as {target_names[1]}: {fp}")
    print(f"• {target_names[1]} correct: {tp}, {target_names[1]} misclassified as {target_names[0]}: {fn}")
    print(f"• {target_names[0]} classification accuracy: {tn/(tn+fp)*100 if (tn+fp) > 0 else 0:.1f}%")
    print(f"• {target_names[1]} classification accuracy: {tp/(tp+fn)*100 if (tp+fn) > 0 else 0:.1f}%")
else:
    print(f"\n📊 Confusion Matrix shape: {cm.shape}")
    print("Confusion matrix interpretation available only for binary classification")

# ⭐️ 추가: 모델 신뢰도 체크
print(f"\n🔍 Model Reliability Check:")
print(f"• Train/Test split ratio: {len(y_train)}/{len(y_test)} ({len(y_test)/(len(y_train)+len(y_test))*100:.1f}% test)")
print(f"• Class balance in test set: {np.bincount(y_test)}")
support_vectors_ratio = len(best_svm_model.support_) / len(y_train) * 100
print(f"• Support vectors: {len(best_svm_model.support_)}/{len(y_train)} ({support_vectors_ratio:.1f}%)")