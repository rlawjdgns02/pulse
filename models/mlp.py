import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# 재현 가능한 결과를 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 1. 데이터셋 불러오기
# 피처(특성)와 라벨 파일을 pandas DataFrame으로 불러옵니다.
features_df = pd.read_csv('features_left.csv')
labels_df = pd.read_csv('labels_left.csv')

# 2. 데이터 준비
# pandas DataFrame을 NumPy 배열로 변환하여 처리합니다.
X = features_df.values
# 기존 라벨은 1과 2로 되어있습니다. 대부분의 라이브러리에서 이진 분류는 0과 1을 표준으로 사용하므로,
# 라벨 값에서 1을 빼서 0과 1로 변환합니다.
y = labels_df.values.flatten() - 1

# 3. MLP 모델 정의 (드롭아웃 추가)
# 과적합을 방지하기 위해 각 은닉층 뒤에 드롭아웃 레이어를 추가합니다.
# 드롭아웃은 훈련 중에 지정된 비율(p=0.5, 즉 50%)의 뉴런을 무작위로 비활성화합니다.
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Dropout(0.2), # 첫 번째 드롭아웃 레이어
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

# 4. 5-겹 교차 검증(5-Fold Cross-Validation) 설정
# 교차 검증은 데이터를 한 번만 나누는 것보다 모델의 성능을 더 안정적으로 평가할 수 있게 해줍니다.
# StratifiedKFold는 각 폴드(fold)의 클래스 비율이 전체 데이터셋의 클래스 비율과 동일하게 유지되도록 합니다.
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 각 폴드의 결과를 저장하여 나중에 분석합니다.
fold_accuracies = []
fold_losses = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []
fold_train_losses = []
fold_train_accuracies = []

# 5. 훈련 및 평가 루프
# 각 폴드를 순회하며 모델을 훈련하고 평가합니다.
for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"--- {fold+1}/{n_splits} 번째 폴드 ---")

    # 현재 폴드에 맞게 데이터를 훈련셋과 검증셋으로 나눕니다.
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # 피처 스케일링: 신경망에서는 데이터 스케일링이 중요합니다.
    # 데이터 유출(data leakage)을 방지하기 위해, 반드시 훈련 데이터에만 `fit`을 적용하고,
    # 그 스케일러를 사용하여 훈련 데이터와 검증 데이터를 모두 변환(`transform`)합니다.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # NumPy 배열을 PyTorch 텐서로 변환합니다.
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

    # 각 폴드마다 모델, 손실 함수, 옵티마이저를 초기화합니다.
    input_size = X_train.shape[1]
    model = MLP(input_size)
    criterion = nn.BCELoss()  # 이진 분류를 위한 Binary Cross-Entropy 손실 함수
    # 옵티마이저에 weight_decay (L2 규제)를 추가하여 가중치 감쇠를 적용합니다.
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) # Adam 옵티마이저

    # 학습률 스케줄러 추가 - 성능이 개선되지 않으면 학습률을 줄입니다
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    # 조기 종료를 위한 변수
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    # 각 에포크의 손실과 정확도를 저장
    train_losses = []
    train_accs = []
    
    # 모델 훈련
    n_epochs = 100
    for epoch in range(n_epochs):
        model.train() # 모델을 훈련 모드로 설정
        optimizer.zero_grad() # 이전 그래디언트 초기화
        outputs = model(X_train_tensor) # 순전파 (Forward pass)
        loss = criterion(outputs, y_train_tensor) # 손실 계산
        loss.backward() # 역전파 (Backward pass), 그래디언트 계산
        optimizer.step() # 가중치 업데이트
        
        # 훈련 정확도 계산
        with torch.no_grad():
            train_preds = outputs.round()
            train_acc = accuracy_score(y_train_tensor.detach().numpy(), train_preds.detach().numpy())
            train_losses.append(loss.item())
            train_accs.append(train_acc)
        
        # 검증 손실 계산 (스케줄러를 위해)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        model.train()
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        
        # 조기 종료 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"조기 종료: {epoch+1} 에포크에서 종료")
                break
        
        # 20 에포크마다 훈련 상황 출력
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}")

    # 검증셋으로 모델 평가
    model.eval() # 모델을 평가 모드로 설정
    with torch.no_grad(): # 평가 시에는 그래디언트 계산을 비활성화
        y_pred = model(X_val_tensor)
        y_pred_class = y_pred.round() # 확률 값을 0 또는 1의 이진 예측값으로 변환
        accuracy = accuracy_score(y_val_tensor, y_pred_class)
        val_loss = criterion(y_pred, y_val_tensor)
        
        # 결과 저장
        fold_accuracies.append(accuracy)
        fold_losses.append(val_loss.item())
        fold_train_losses.append(train_losses)
        fold_train_accuracies.append(train_accs)
        
        # 전체 예측 결과 저장 (나중에 전체 confusion matrix를 위해)
        all_y_true.extend(y_val_tensor.numpy().flatten())
        all_y_pred.extend(y_pred_class.numpy().flatten())
        all_y_pred_proba.extend(y_pred.numpy().flatten())
        
        print(f"-> {fold+1}번째 폴드 최종 검증 정확도: {accuracy:.4f}, 검증 손실: {val_loss.item():.4f}\n")

# 6. 최종 결과 및 시각화
print("\n=== Cross-Validation Results Summary ===\n")
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
mean_loss = np.mean(fold_losses)
std_loss = np.std(fold_losses)

print(f"Average Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Average Loss: {mean_loss:.4f} ± {std_loss:.4f}")
print(f"Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")

# 전체 데이터에 대한 분류 보고서
print("\n=== Overall Classification Report ===")
print(classification_report(all_y_true, all_y_pred, target_names=['Class 0 (Cold)', 'Class 1 (Hot)']))

# 시각화 시작 - 한글 폰트 설정
import matplotlib.font_manager as fm

# 여러 폰트 옵션 시도 (Windows 한글 폰트들)
font_candidates = ['Malgun Gothic', 'Arial Unicode MS', 'Gulim', 'Dotum', 'Batang']
available_fonts = [f.name for f in fm.fontManager.ttflist]

# 사용 가능한 첫 번째 한글 폰트 선택
selected_font = 'DejaVu Sans'  # 기본값
for font in font_candidates:
    if font in available_fonts:
        selected_font = font
        break

print(f"사용할 폰트: {selected_font}")

plt.rcParams.update({
    'font.family': selected_font,
    'font.sans-serif': [selected_font, 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.autolayout': True
})
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('MLP Model Performance Analysis', fontsize=16, fontweight='bold')

# 1. 폴드별 정확도 비교
axes[0, 0].bar(range(1, n_splits+1), fold_accuracies, alpha=0.7, color='skyblue', edgecolor='navy')
axes[0, 0].axhline(y=mean_accuracy, color='red', linestyle='--', label=f'Average: {mean_accuracy:.4f}')
axes[0, 0].set_xlabel('Fold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Validation Accuracy by Fold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Confusion Matrix
cm = confusion_matrix(all_y_true, all_y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=['Cold(0)', 'Hot(1)'], yticklabels=['Cold(0)', 'Hot(1)'])
axes[0, 1].set_title('Overall Confusion Matrix')
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')

# 3. ROC Curve
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_proba)
roc_auc = auc(fpr, tpr)
axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1, 0].set_xlim([0.0, 1.0])
axes[1, 0].set_ylim([0.0, 1.05])
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 훈련 과정 (첫 번째 폴드 예시)
if fold_train_losses:
    first_fold_losses = fold_train_losses[0]
    first_fold_accs = fold_train_accuracies[0]
    epochs_range = range(1, len(first_fold_losses) + 1)
    
    ax4_twin = axes[1, 1].twinx()
    line1 = axes[1, 1].plot(epochs_range, first_fold_losses, 'b-', label='Training Loss', alpha=0.7)
    line2 = ax4_twin.plot(epochs_range, first_fold_accs, 'r-', label='Training Accuracy', alpha=0.7)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss', color='b')
    ax4_twin.set_ylabel('Accuracy', color='r')
    axes[1, 1].set_title('Training Process (Fold 1)')
    
    # 범례 결합
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    axes[1, 1].legend(lines, labels, loc='center right')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('mlp_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'mlp_performance_analysis.png'.")