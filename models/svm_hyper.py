import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== SVM Multi-Kernel Hyperparameter Grid Search ===")

# 1. 데이터 로드 및 전처리
X = pd.read_csv('features_left.csv')
y = pd.read_csv('labels_left.csv').astype(int)
y_flat = np.ravel(y)

print(f"데이터 크기: {X.shape}")
print(f"레이블 분포: {np.bincount(y_flat)}")

# 2. Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

# 3. 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 커널별 파라미터 그리드 정의
param_grids = [
    # Linear kernel (gamma 불필요)
    {
        'kernel': ['linear'], 
        'C': [0.1, 1, 10, 100, 1000]
    },
    
    # RBF kernel
    {
        'kernel': ['rbf'], 
        'C': [0.1, 1, 10, 100, 1000], 
        'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001]
    },
    
    # Polynomial kernel
    {
        'kernel': ['poly'], 
        'C': [0.1, 1, 10, 100], 
        'gamma': ['scale', 'auto', 1, 0.1, 0.01],
        'degree': [2, 3, 4]
    },
    
    # Sigmoid kernel
    {
        'kernel': ['sigmoid'], 
        'C': [0.1, 1, 10, 100], 
        'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001]
    }
]

# 5. 평가 지표 설정
scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'precision_macro': 'precision_macro'}

# 6. GridSearchCV 실행
print("\n--- Grid Search 실행 중 ---")
grid_search = GridSearchCV(
    SVC(random_state=42), 
    param_grids, 
    cv=5, 
    scoring=scoring, 
    refit='f1_macro',  # f1_macro로 최적 모델 선택
    n_jobs=-1, 
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

# 7. 결과 분석
print("\n--- Grid Search 결과 ---")
print(f"최적의 하이퍼파라미터: {grid_search.best_params_}")
print(f"최고 교차 검증 F1-Score: {grid_search.best_score_:.4f}")

# 테스트 데이터로 최종 성능 평가
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test_scaled, y_test)
y_pred = best_model.predict(X_test_scaled)

print(f"테스트 데이터 정확도: {test_accuracy:.4f}")

# 8. 커널별 최고 성능 비교
results_df = pd.DataFrame(grid_search.cv_results_)

print("\n--- 커널별 최고 성능 비교 ---")
kernel_performance = {}

for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    kernel_results = results_df[results_df['param_kernel'] == kernel]
    if len(kernel_results) > 0:
        best_idx = kernel_results['mean_test_f1_macro'].idxmax()
        best_result = results_df.loc[best_idx]
        
        kernel_performance[kernel] = {
            'f1_score': best_result['mean_test_f1_macro'],
            'accuracy': best_result['mean_test_accuracy'],
            'precision': best_result['mean_test_precision_macro'],
            'params': {k.replace('param_', ''): v for k, v in best_result.items() 
                      if k.startswith('param_')}
        }
        
        print(f"\n{kernel.upper()} 커널:")
        print(f"  F1-Score: {best_result['mean_test_f1_macro']:.4f}")
        print(f"  정확도: {best_result['mean_test_accuracy']:.4f}")
        print(f"  정밀도: {best_result['mean_test_precision_macro']:.4f}")
        print(f"  파라미터: {kernel_performance[kernel]['params']}")

# 9. 상세 분류 보고서
print("\n--- 테스트 데이터 분류 보고서 ---")
print(classification_report(y_test, y_pred))

# 10. 혼동 행렬
print("\n--- 혼동 행렬 ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 11. 시각화 함수들
def plot_kernel_comparison(kernel_performance):
    """커널별 성능 비교 막대그래프"""
    kernels = list(kernel_performance.keys())
    f1_scores = [kernel_performance[k]['f1_score'] for k in kernels]
    accuracies = [kernel_performance[k]['accuracy'] for k in kernels]
    precisions = [kernel_performance[k]['precision'] for k in kernels]
    
    x = np.arange(len(kernels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, f1_scores, width, label='F1-Score', alpha=0.8)
    ax.bar(x, accuracies, width, label='Accuracy', alpha=0.8)
    ax.bar(x + width, precisions, width, label='Precision', alpha=0.8)
    
    ax.set_xlabel('커널 유형')
    ax.set_ylabel('성능 점수')
    ax.set_title('SVM 커널별 성능 비교')
    ax.set_xticks(x)
    ax.set_xticklabels(kernels)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 수치 표시
    for i, (f1, acc, prec) in enumerate(zip(f1_scores, accuracies, precisions)):
        ax.text(i - width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom')
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        ax.text(i + width, prec + 0.01, f'{prec:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('svm_kernel_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm, classes=['한체질', '열체질']):
    """혼동 행렬 히트맵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('예측값')
    plt.ylabel('실제값')
    plt.title('혼동 행렬')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_parameter_heatmap_by_kernel(results_df, kernel):
    """특정 커널의 파라미터별 성능 히트맵"""
    kernel_results = results_df[results_df['param_kernel'] == kernel].copy()
    
    if len(kernel_results) == 0:
        print(f"{kernel} 커널 결과가 없습니다.")
        return
    
    try:
        if kernel == 'linear':
            # Linear는 C만 있음
            plt.figure(figsize=(10, 4))
            c_values = sorted(kernel_results['param_C'].unique())
            f1_scores = []
            
            for c in c_values:
                score = kernel_results[kernel_results['param_C'] == c]['mean_test_f1_macro'].iloc[0]
                f1_scores.append(score)
            
            plt.bar(range(len(c_values)), f1_scores, alpha=0.7, color='skyblue')
            plt.xlabel('C 값')
            plt.ylabel('F1-Score')
            plt.title(f'{kernel.upper()} 커널 파라미터별 성능')
            plt.xticks(range(len(c_values)), [str(c) for c in c_values])
            
            # 수치 표시
            for i, score in enumerate(f1_scores):
                plt.text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom')
            
            plt.grid(True, alpha=0.3)
            
        elif kernel in ['rbf', 'sigmoid']:
            # C와 gamma가 있는 커널
            pivot_table = kernel_results.pivot_table(
                values='mean_test_f1_macro',
                index='param_gamma',
                columns='param_C',
                aggfunc='mean'
            )
            
            # gamma 값 정렬 문제 해결: 문자열과 숫자 분리해서 처리
            gamma_order = []
            string_gammas = []
            numeric_gammas = []
            
            for gamma in pivot_table.index:
                if isinstance(gamma, str):
                    string_gammas.append(gamma)
                else:
                    numeric_gammas.append(gamma)
            
            # 문자열 gamma를 앞에, 숫자 gamma를 큰 값부터 정렬
            string_gammas.sort()
            numeric_gammas.sort(reverse=True)
            gamma_order = string_gammas + numeric_gammas
            
            # 정렬된 순서로 인덱스 재배열
            pivot_table = pivot_table.reindex(gamma_order)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis',
                       cbar_kws={'label': 'F1-Score'})
            plt.xlabel('C 값')
            plt.ylabel('Gamma 값')
            plt.title(f'{kernel.upper()} 커널 하이퍼파라미터 히트맵')
            
        elif kernel == 'poly':
            # Polynomial은 3차원이므로 degree별로 분리해서 표시
            degrees = sorted(kernel_results['param_degree'].unique())
            
            fig, axes = plt.subplots(1, len(degrees), figsize=(6*len(degrees), 5))
            if len(degrees) == 1:
                axes = [axes]
            
            for i, degree in enumerate(degrees):
                degree_results = kernel_results[kernel_results['param_degree'] == degree]
                pivot_table = degree_results.pivot_table(
                    values='mean_test_f1_macro',
                    index='param_gamma',
                    columns='param_C',
                    aggfunc='mean'
                )
                
                # gamma 정렬 문제 해결
                gamma_order = []
                string_gammas = []
                numeric_gammas = []
                
                for gamma in pivot_table.index:
                    if isinstance(gamma, str):
                        string_gammas.append(gamma)
                    else:
                        numeric_gammas.append(gamma)
                
                string_gammas.sort()
                numeric_gammas.sort(reverse=True)
                gamma_order = string_gammas + numeric_gammas
                pivot_table = pivot_table.reindex(gamma_order)
                
                sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='viridis',
                           ax=axes[i], cbar_kws={'label': 'F1-Score'})
                axes[i].set_title(f'Degree = {int(degree)}')
                axes[i].set_xlabel('C 값')
                axes[i].set_ylabel('Gamma 값')
            
            plt.suptitle(f'{kernel.upper()} 커널 하이퍼파라미터 히트맵', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'svm_{kernel}_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"{kernel} 커널 히트맵 생성 중 오류: {e}")
        print("히트맵 생성을 건너뜁니다.")

# 12. 시각화 실행
print("\n--- 시각화 생성 중 ---")

# 커널 비교 차트
plot_kernel_comparison(kernel_performance)

# 혼동 행렬
plot_confusion_matrix(cm)

# 각 커널별 파라미터 히트맵
for kernel in ['linear', 'rbf', 'sigmoid']:
    plot_parameter_heatmap_by_kernel(results_df, kernel)

# 13. 최종 추천
print("\n--- 최종 추천 ---")
best_kernel = grid_search.best_params_['kernel']
print(f"🏆 최고 성능 커널: {best_kernel.upper()}")
print(f"🎯 최적 파라미터: {grid_search.best_params_}")
print(f"📊 CV F1-Score: {grid_search.best_score_:.4f}")
print(f"📈 테스트 정확도: {test_accuracy:.4f}")

# 성능 순위
performance_ranking = sorted(kernel_performance.items(), 
                           key=lambda x: x[1]['f1_score'], reverse=True)
print(f"\n커널 성능 순위:")
for i, (kernel, perf) in enumerate(performance_ranking, 1):
    print(f"  {i}. {kernel.upper()}: F1={perf['f1_score']:.4f}")

print(f"\n모든 그래프가 저장되었습니다!")