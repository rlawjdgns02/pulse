import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== LogisticRegression 하이퍼파라미터 최적화 및 시각화 ===")

# 데이터 로드 (39개 특성으로 된 CSV 파일)
X = pd.read_csv('features_left.csv')  # 39개 특성 파일 경로에 맞게 수정
y = pd.read_csv('labels_left.csv')  # 레이블 파일 경로에 맞게 수정
y = y.astype(int)

# 라벨 처리
y_flat = np.ravel(y)
if set(np.unique(y_flat)) == {1, 2}:
    y_flat = y_flat - 1
    print("라벨 변환: {1,2} → {0,1}")

print(f"데이터 크기: {X.shape}")
print(f"클래스 분포: {np.bincount(y_flat)}")

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_flat, test_size=0.2, random_state=42, stratify=y_flat
)

# 데이터 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"훈련 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# 1. 포괄적인 하이퍼파라미터 그리드 정의
print(f"\n1. 하이퍼파라미터 그리드 설정")
print("-" * 50)

# 각 solver별로 지원하는 penalty와 조합
param_grid = [
    # liblinear solver (L1, L2 지원)
    {
        'solver': ['liblinear'],
        'penalty': ['l1', 'l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [1000, 2000]
    },
    # saga solver (L1, L2, elasticnet 지원)
    {
        'solver': ['saga'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # elasticnet용
        'max_iter': [1000, 2000, 5000]
    },
    # lbfgs solver (L2만 지원, 기본값)
    {
        'solver': ['lbfgs'],
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [1000, 2000, 5000]
    }
]

total_combinations = 0
for grid in param_grid:
    combinations = 1
    for key, values in grid.items():
        if key == 'l1_ratio' and 'elasticnet' in grid.get('penalty', []):
            combinations *= len(values)
        elif key != 'l1_ratio':
            combinations *= len(values)
    total_combinations += combinations

print(f"총 하이퍼파라미터 조합 수: {total_combinations}개")

# 2. GridSearchCV 실행
print(f"\n2. GridSearchCV 실행")
print("-" * 50)

grid_search = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("GridSearchCV 실행 중...")
grid_search.fit(X_train_scaled, y_train)

print(f"\n최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.1f}%)")

# 테스트 성능
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"테스트 정확도: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")
print(f"CV vs 테스트 차이: {abs(grid_search.best_score_ - test_accuracy)*100:.1f}%p")

# 3. 결과 분석 및 시각화
print(f"\n3. 결과 분석 및 시각화")
print("-" * 50)

results_df = pd.DataFrame(grid_search.cv_results_)

# 상위 10개 결과
top_10 = results_df.nlargest(10, 'mean_test_score')
print("\n상위 10개 하이퍼파라미터 조합:")
for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    params_str = ', '.join([f"{k.replace('param_', '')}: {v}" for k, v in row.items() 
                           if k.startswith('param_')])
    print(f"{i:2d}. 점수: {row['mean_test_score']:.4f} | {params_str}")

# 4. 격자판 시각화
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('LogisticRegression 하이퍼파라미터 최적화 결과', fontsize=16, fontweight='bold')

# 4-1. Solver별 성능 비교
ax1 = axes[0, 0]
solver_performance = results_df.groupby('param_solver')['mean_test_score'].agg(['mean', 'max', 'std'])
solver_names = solver_performance.index
x_pos = np.arange(len(solver_names))

bars = ax1.bar(x_pos, solver_performance['mean'], 
               yerr=solver_performance['std'], capsize=5, alpha=0.7, color='skyblue')
ax1.set_xlabel('Solver')
ax1.set_ylabel('평균 CV 점수')
ax1.set_title('Solver별 성능 비교')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(solver_names)

# 막대 위에 수치 표시
for i, (bar, mean_val, max_val) in enumerate(zip(bars, solver_performance['mean'], solver_performance['max'])):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'평균: {mean_val:.3f}\n최고: {max_val:.3f}', ha='center', va='bottom', fontsize=9)

# 4-2. C 값에 따른 성능 변화 (penalty별)
ax2 = axes[0, 1]
for penalty in ['l1', 'l2']:
    penalty_data = results_df[results_df['param_penalty'] == penalty]
    if len(penalty_data) > 0:
        c_performance = penalty_data.groupby('param_C')['mean_test_score'].mean()
        ax2.plot(range(len(c_performance)), c_performance.values, marker='o', label=f'{penalty}')

ax2.set_xlabel('C 값 (로그 스케일)')
ax2.set_ylabel('평균 CV 점수')
ax2.set_title('C 값에 따른 성능 변화 (Penalty별)')
ax2.set_xticks(range(len(c_performance)))
ax2.set_xticklabels([f'{c:.3f}' for c in c_performance.index], rotation=45)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 4-3. Penalty별 성능 분포
ax3 = axes[0, 2]
penalty_types = results_df['param_penalty'].dropna().unique()
penalty_scores = [results_df[results_df['param_penalty'] == p]['mean_test_score'].values 
                 for p in penalty_types]

box_plot = ax3.boxplot(penalty_scores, labels=penalty_types, patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(box_plot['boxes'], colors[:len(penalty_types)]):
    patch.set_facecolor(color)

ax3.set_xlabel('Penalty 유형')
ax3.set_ylabel('CV 점수')
ax3.set_title('Penalty별 성능 분포')
ax3.grid(True, alpha=0.3)

# 4-4. C vs Solver 히트맵 (L2 penalty만)
ax4 = axes[1, 0]
l2_data = results_df[results_df['param_penalty'] == 'l2']
if len(l2_data) > 0:
    heatmap_data = l2_data.pivot_table(
        values='mean_test_score', 
        index='param_solver', 
        columns='param_C', 
        aggfunc='mean'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', 
                ax=ax4, cbar_kws={'label': 'CV 점수'})
    ax4.set_title('C vs Solver 성능 히트맵 (L2 Penalty)')
    ax4.set_xlabel('C 값')
    ax4.set_ylabel('Solver')

# 4-5. 성능 향상 추이
ax5 = axes[1, 1]
sorted_results = results_df.sort_values('mean_test_score', ascending=False)
top_n = min(50, len(sorted_results))
ax5.plot(range(1, top_n+1), sorted_results['mean_test_score'].head(top_n), 
         marker='o', markersize=3, alpha=0.7)
ax5.axhline(y=grid_search.best_score_, color='red', linestyle='--', 
           label=f'최고 점수: {grid_search.best_score_:.4f}')
ax5.set_xlabel('순위')
ax5.set_ylabel('CV 점수')
ax5.set_title(f'상위 {top_n}개 조합 성능 분포')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 4-6. 과적합 분석 (Train vs Validation 점수)
ax6 = axes[1, 2]
train_scores = results_df['mean_train_score']
test_scores = results_df['mean_test_score']

ax6.scatter(train_scores, test_scores, alpha=0.6, s=20)
ax6.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='완벽한 일치선')

# 최고 성능 점 강조
best_idx = results_df['mean_test_score'].idxmax()
best_train = results_df.loc[best_idx, 'mean_train_score']
best_test = results_df.loc[best_idx, 'mean_test_score']
ax6.scatter(best_train, best_test, color='red', s=100, marker='*', 
           label=f'최고 성능\n({best_train:.3f}, {best_test:.3f})')

ax6.set_xlabel('Train CV 점수')
ax6.set_ylabel('Test CV 점수')
ax6.set_title('과적합 분석 (Train vs Test)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Validation Curve - C 파라미터 세부 분석
print(f"\n4. C 파라미터 세부 분석")
print("-" * 50)

# 최적 solver와 penalty로 C 값만 변경하며 분석
best_solver = grid_search.best_params_['solver']
best_penalty = grid_search.best_params_['penalty']

# ElasticNet의 경우 l1_ratio도 필요
if best_penalty == 'elasticnet':
    best_l1_ratio = grid_search.best_params_['l1_ratio']
    best_max_iter = grid_search.best_params_['max_iter']
    
    model_for_validation = LogisticRegression(
        solver=best_solver, 
        penalty=best_penalty, 
        l1_ratio=best_l1_ratio,
        max_iter=best_max_iter,
        random_state=42
    )
    print(f"분석 모델: solver={best_solver}, penalty={best_penalty}, l1_ratio={best_l1_ratio}")
else:
    model_for_validation = LogisticRegression(
        solver=best_solver, 
        penalty=best_penalty, 
        random_state=42
    )
    print(f"분석 모델: solver={best_solver}, penalty={best_penalty}")

C_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

try:
    train_scores_c, validation_scores_c = validation_curve(
        model_for_validation, X_train_scaled, y_train, 
        param_name='C', param_range=C_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    # Validation Curve 시각화
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    train_mean = np.mean(train_scores_c, axis=1)
    train_std = np.std(train_scores_c, axis=1)
    validation_mean = np.mean(validation_scores_c, axis=1)
    validation_std = np.std(validation_scores_c, axis=1)

    plt.semilogx(C_range, train_mean, 'o-', color='blue', label='훈련 점수')
    plt.fill_between(C_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')

    plt.semilogx(C_range, validation_mean, 'o-', color='red', label='검증 점수')
    plt.fill_between(C_range, validation_mean - validation_std, validation_mean + validation_std, alpha=0.1, color='red')

    plt.xlabel('C (정규화 강도)')
    plt.ylabel('정확도')
    plt.title(f'Validation Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # C 값별 성능 표 - 간단한 버전
    plt.subplot(1, 2, 2)
    
    # 텍스트로 표 만들기
    table_text = "C 값별 성능 비교\n" + "="*30 + "\n"
    table_text += f"{'C':<8} {'Train':<7} {'Valid':<7} {'Gap':<6}\n"
    table_text += "-"*30 + "\n"
    
    for i, c in enumerate(C_range):
        train_score = train_mean[i]
        valid_score = validation_mean[i]
        gap = train_score - valid_score
        table_text += f"{c:<8.3f} {train_score:<7.3f} {valid_score:<7.3f} {gap:<6.3f}\n"
    
    plt.text(0.1, 0.9, table_text, transform=plt.gca().transAxes, 
             fontfamily='monospace', fontsize=10, verticalalignment='top')
    plt.axis('off')
    plt.title('C 값별 성능 상세표')

    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Validation curve 생성 중 오류: {e}")
    print("대신 간단한 C 값 비교를 수행합니다.")
    
    # 대안: 간단한 C 값 비교
    plt.figure(figsize=(10, 6))
    
    c_scores = []
    for c in C_range:
        if best_penalty == 'elasticnet':
            temp_model = LogisticRegression(
                C=c, solver=best_solver, penalty=best_penalty, 
                l1_ratio=best_l1_ratio, max_iter=best_max_iter, random_state=42
            )
        else:
            temp_model = LogisticRegression(
                C=c, solver=best_solver, penalty=best_penalty, random_state=42
            )
        
        scores = cross_val_score(temp_model, X_train_scaled, y_train, cv=5)
        c_scores.append(scores.mean())
    
    plt.semilogx(C_range, c_scores, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('C 값')
    plt.ylabel('5-Fold CV 정확도')
    plt.title('C 파라미터에 따른 성능 변화')
    plt.grid(True, alpha=0.3)
    
    # 최고 성능 지점 표시
    best_c_idx = np.argmax(c_scores)
    plt.scatter(C_range[best_c_idx], c_scores[best_c_idx], 
               color='red', s=100, zorder=5, label=f'최고 성능: C={C_range[best_c_idx]}')
    plt.legend()
    
    for i, (c, score) in enumerate(zip(C_range, c_scores)):
        plt.annotate(f'{score:.3f}', (c, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# 6. 최종 결과 요약
print(f"\n5. 최종 결과 요약")
print("=" * 50)

print(f"최적 하이퍼파라미터:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\n성능 결과:")
print(f"  최적 CV 점수: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.1f}%)")
print(f"  테스트 점수: {test_accuracy:.4f} ({test_accuracy*100:.1f}%)")

# 유전 알고리즘 결과와 비교
genetic_score = 0.7406
improvement = grid_search.best_score_ - genetic_score
print(f"\n유전 알고리즘과 비교:")
print(f"  유전 알고리즘 (기본 파라미터): {genetic_score:.4f} ({genetic_score*100:.1f}%)")
print(f"  하이퍼파라미터 최적화 후: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.1f}%)")
print(f"  개선 정도: {improvement:.4f} ({improvement*100:.1f}%p)")

if improvement > 0.02:
    print("  >> 하이퍼파라미터 최적화로 유의미한 개선")
elif improvement > 0:
    print("  >> 약간의 개선")
else:
    print("  >> 개선 효과 미미 또는 악화")

# 목표와 비교
target_score = 0.80
gap_to_target = target_score - max(grid_search.best_score_, test_accuracy)
print(f"\n80% 목표와 비교:")
print(f"  목표까지 남은 격차: {gap_to_target*100:.1f}%p")

if gap_to_target <= 0:
    print("  🎯 목표 달성!")
elif gap_to_target <= 0.05:
    print("  📈 목표에 매우 근접")
elif gap_to_target <= 0.1:
    print("  📊 목표에 근접하지만 추가 개선 필요")
else:
    print("  📉 목표 달성을 위해 다른 접근법 필요")