import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest, 
    VarianceThreshold, RFECV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("좌수/우수 분리 특성선택 + 하이퍼파라미터 최적화 + 비교분석")
print("=" * 80)

# 1. 데이터 로드
print("\n1. 데이터 로드")
left_features = pd.read_csv('data/all_features_left.csv')
left_labels = pd.read_csv('data/labels_all_left.csv')
right_features = pd.read_csv('data/all_features_right.csv')
right_labels = pd.read_csv('data/labels_all_right.csv')

print(f"좌수: {left_features.shape}, 우수: {right_features.shape}")

# 2. 4단계 특성 선택 함수
def perform_feature_selection(features, labels, name):
    """4단계 특성 선택"""
    print(f"\n{'='*60}")
    print(f"STEP 1: {name} 특성 선택")
    print(f"{'='*60}")
    
    X, y = features.copy(), labels.iloc[:, 0].copy()
    print(f"원본: {X.shape}, 레이블 분포: {y.value_counts().to_dict()}")
    
    # 1단계: 분산 + 상관관계 필터링
    variance_filter = VarianceThreshold(threshold=0.01)
    X_step1a = variance_filter.fit_transform(X)
    X_step1 = pd.DataFrame(X_step1a, columns=X.columns[variance_filter.get_support()])
    
    # 상관관계 제거
    corr_matrix = X_step1.corr()
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_features.add(corr_matrix.columns[j])
    
    X_step1_final = X_step1.drop(columns=high_corr_features)
    print(f"1단계: {X.shape[1]} → {X_step1_final.shape[1]}개")
    
    # 2단계: 비선형 특성 선택 (F-score + Mutual Information)
    print(f"2단계: 비선형 특성 선택")
    print("-" * 30)
    
    # F-score (선형)
    f_scores, p_values = f_classif(X_step1_final, y)
    f_series = pd.Series(f_scores, index=X_step1_final.columns)
    
    # Mutual Information (비선형)
    mi_scores = mutual_info_classif(X_step1_final, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X_step1_final.columns)
    
    # 순위 결합
    f_ranking = f_series.rank(ascending=False)
    mi_ranking = mi_series.rank(ascending=False)
    combined_ranking = mi_ranking
    combined_ranking = combined_ranking.sort_values()
    
    print(f"   F-score 1위: {f_series.idxmax()} (점수: {f_series.max():.3f})")
    print(f"   MI 1위: {mi_series.idxmax()} (점수: {mi_series.max():.3f})")
    print(f"   결합 1위: {combined_ranking.index[0]}")
    
    # 통계적 유의성 고려
    significant_features = X_step1_final.columns[p_values < 0.05]
    print(f"   통계적 유의 특성: {len(significant_features)}개")
    
    if len(significant_features) >= 8:
        significant_ranking = combined_ranking[significant_features]
        selected_features = significant_ranking.head(min(20, len(significant_features))).index.tolist()
        print(f"   → 유의한 특성 중 상위 {len(selected_features)}개 선택")
    else:
        selected_features = combined_ranking.head(15).index.tolist()
        print(f"   → 전체 특성 중 상위 {len(selected_features)}개 선택")
    
    X_step2 = X_step1_final[selected_features]
    print(f"2단계: {X_step1_final.shape[1]} → {X_step2.shape[1]}개")
    
    # 3단계: L1 정규화
    scaler = StandardScaler()
    X_step2_scaled = scaler.fit_transform(X_step2)
    
    best_result = None
    best_score = 0
    
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        lasso = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42)
        lasso.fit(X_step2_scaled, y)
        
        selected_mask = lasso.coef_[0] != 0
        if np.sum(selected_mask) > 0:
            cv_score = cross_val_score(lasso, X_step2_scaled, y, cv=5).mean()
            if cv_score > best_score:
                best_score = cv_score
                best_result = {'C': C, 'features': X_step2.columns[selected_mask].tolist()}
    
    X_step3 = X_step2[best_result['features']] if best_result else X_step2
    print(f"3단계: {X_step2.shape[1]} → {X_step3.shape[1]}개")
    
    # 4단계: Random Forest RFECV (비선형)
    print(f"4단계: 비선형 RFECV (Random Forest)")
    print("-" * 30)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    min_features = max(2, X_step3.shape[1] // 4)
    
    rfecv = RFECV(rf, step=1, cv=5, scoring='accuracy',
                  min_features_to_select=min_features, n_jobs=-1)
    
    X_step3_scaled = StandardScaler().fit_transform(X_step3)
    rfecv.fit(X_step3_scaled, y)
    
    X_final = X_step3.iloc[:, rfecv.support_]
    final_features = X_step3.columns[rfecv.support_]
    
    print(f"4단계: {X_step3.shape[1]} → {X_final.shape[1]}개")
    print(f"{name} 비선형 특성선택 완료: {len(final_features)}개 특성")
    
    return X_final, y, final_features.tolist()

# 3. 하이퍼파라미터 그리드서치 함수
def perform_grid_search(X, y, name):
    """하이퍼파라미터 그리드서치 + 시각화"""
    print(f"\n{'='*60}")
    print(f"STEP 2: {name} 하이퍼파라미터 그리드서치")
    print(f"{'='*60}")
    
    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 커널별 파라미터 그리드
    param_grids = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100, 1000], 
         'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.0001]},
        {'kernel': ['poly'], 'C': [0.1, 1, 10, 100], 
         'gamma': ['scale', 'auto', 1, 0.1, 0.01], 'degree': [2, 3, 4]},
        {'kernel': ['sigmoid'], 'C': [0.1, 1, 10, 100], 
         'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001]}
    ]
    
    scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro', 'precision_macro': 'precision_macro'}
    
    print("그리드서치 실행중...")
    grid_search = GridSearchCV(
        SVC(random_state=42), param_grids, cv=5, 
        scoring=scoring, refit='f1_macro', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # 테스트 성능
    test_accuracy = grid_search.best_estimator_.score(X_test_scaled, y_test)
    
    print(f"최적 파라미터: {grid_search.best_params_}")
    print(f"CV F1-Score: {grid_search.best_score_:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    
    # 커널별 성능 분석
    results_df = pd.DataFrame(grid_search.cv_results_)
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
    
    return {
        'name': name,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'test_accuracy': test_accuracy,
        'kernel_performance': kernel_performance,
        'results_df': results_df
    }

# 4. 시각화 함수
def create_grid_search_heatmap(results_df, name):
    """그리드서치 결과 히트맵"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{name} 하이퍼파라미터 그리드서치 결과', fontsize=16, fontweight='bold')
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    axes = axes.flatten()
    
    for i, kernel in enumerate(kernels):
        kernel_results = results_df[results_df['param_kernel'] == kernel]
        
        if len(kernel_results) == 0:
            axes[i].text(0.5, 0.5, f'{kernel} 결과 없음', ha='center', va='center', 
                        transform=axes[i].transAxes)
            axes[i].set_title(f'{kernel.upper()} 커널')
            continue
        
        if kernel == 'linear':
            c_values = sorted(kernel_results['param_C'].unique())
            f1_scores = []
            
            for c in c_values:
                score = kernel_results[kernel_results['param_C'] == c]['mean_test_f1_macro'].iloc[0]
                f1_scores.append(score)
            
            axes[i].bar(range(len(c_values)), f1_scores, alpha=0.7, color='skyblue')
            axes[i].set_xticks(range(len(c_values)))
            axes[i].set_xticklabels([str(c) for c in c_values])
            axes[i].set_xlabel('C 값')
            axes[i].set_ylabel('F1-Score')
            
            for j, score in enumerate(f1_scores):
                axes[i].text(j, score + 0.005, f'{score:.3f}', ha='center', va='bottom')
        
        elif kernel in ['rbf', 'sigmoid']:
            try:
                pivot_table = kernel_results.pivot_table(
                    values='mean_test_f1_macro',
                    index='param_gamma',
                    columns='param_C',
                    aggfunc='mean'
                )
                
                string_gammas = [g for g in pivot_table.index if isinstance(g, str)]
                numeric_gammas = [g for g in pivot_table.index if not isinstance(g, str)]
                gamma_order = sorted(string_gammas) + sorted(numeric_gammas, reverse=True)
                pivot_table = pivot_table.reindex(gamma_order)
                
                sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis',
                           ax=axes[i], cbar_kws={'label': 'F1-Score'})
                axes[i].set_xlabel('C 값')
                axes[i].set_ylabel('Gamma 값')
            except:
                axes[i].text(0.5, 0.5, f'{kernel} 히트맵 생성 실패', ha='center', va='center',
                            transform=axes[i].transAxes)
        
        else:  # poly
            degrees = sorted(kernel_results['param_degree'].unique())
            if len(degrees) == 1:
                degree = degrees[0]
                degree_results = kernel_results[kernel_results['param_degree'] == degree]
                try:
                    pivot_table = degree_results.pivot_table(
                        values='mean_test_f1_macro',
                        index='param_gamma',
                        columns='param_C',
                        aggfunc='mean'
                    )
                    
                    string_gammas = [g for g in pivot_table.index if isinstance(g, str)]
                    numeric_gammas = [g for g in pivot_table.index if not isinstance(g, str)]
                    gamma_order = sorted(string_gammas) + sorted(numeric_gammas, reverse=True)
                    pivot_table = pivot_table.reindex(gamma_order)
                    
                    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis',
                               ax=axes[i], cbar_kws={'label': 'F1-Score'})
                    axes[i].set_xlabel('C 값')
                    axes[i].set_ylabel('Gamma 값')
                    axes[i].set_title(f'{kernel.upper()} (degree={degree})')
                except:
                    axes[i].text(0.5, 0.5, f'{kernel} 히트맵 생성 실패', ha='center', va='center',
                                transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, f'{kernel}\n다중 degree', ha='center', va='center',
                            transform=axes[i].transAxes)
        
        if 'set_title' not in locals() or kernel != 'poly':
            axes[i].set_title(f'{kernel.upper()} 커널')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{name}_gridsearch_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. 실행: 좌수 분석
print("\n" + "="*80)
print("좌수 데이터 분석 시작")
print("="*80)

left_X, left_y, left_features = perform_feature_selection(left_features, left_labels, "좌수")
left_grid_results = perform_grid_search(left_X, left_y, "좌수")

create_grid_search_heatmap(left_grid_results['results_df'], "좌수")

# 6. 실행: 우수 분석  
print("\n" + "="*80)
print("우수 데이터 분석 시작")
print("="*80)

right_X, right_y, right_features = perform_feature_selection(right_features, right_labels, "우수")
right_grid_results = perform_grid_search(right_X, right_y, "우수")

create_grid_search_heatmap(right_grid_results['results_df'], "우수")

# 7. STEP 3: 최종 비교 및 정리
print("\n" + "="*80)
print("STEP 3: 최종 결과 비교 및 정리")
print("="*80)

print(f"\n📋 선택된 특성 리스트")
print("-" * 60)

print(f"\n좌수 특성 ({len(left_features)}개):")
for i, feature in enumerate(left_features, 1):
    print(f"  {i:2d}. {feature}")

print(f"\n우수 특성 ({len(right_features)}개):")  
for i, feature in enumerate(right_features, 1):
    print(f"  {i:2d}. {feature}")

common_features = set(left_features) & set(right_features)
print(f"\n공통 특성 ({len(common_features)}개):")
if common_features:
    for i, feature in enumerate(sorted(common_features), 1):
        print(f"  {i:2d}. {feature}")
else:
    print("  없음")

print(f"\n⚙️ 최적 하이퍼파라미터 비교표")
print("-" * 80)

comparison_table = pd.DataFrame({
    '항목': ['데이터', '특성개수', '최적커널', 'C값', 'Gamma', 'Degree', 'CV F1-Score', '테스트정확도'],
    '좌수': [
        '좌수',
        len(left_features),
        left_grid_results['best_params']['kernel'],
        left_grid_results['best_params']['C'],
        left_grid_results['best_params'].get('gamma', 'N/A'),
        left_grid_results['best_params'].get('degree', 'N/A'),
        f"{left_grid_results['best_score']:.4f}",
        f"{left_grid_results['test_accuracy']:.4f}"
    ],
    '우수': [
        '우수',
        len(right_features),
        right_grid_results['best_params']['kernel'],
        right_grid_results['best_params']['C'],
        right_grid_results['best_params'].get('gamma', 'N/A'),
        right_grid_results['best_params'].get('degree', 'N/A'),
        f"{right_grid_results['best_score']:.4f}",
        f"{right_grid_results['test_accuracy']:.4f}"
    ]
})

print(comparison_table.to_string(index=False))

def create_final_comparison():
    """최종 비교 시각화"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('좌수 vs 우수 종합 비교', fontsize=16, fontweight='bold')
    
    metrics = ['CV F1-Score', '테스트 정확도']
    left_scores = [left_grid_results['best_score'], left_grid_results['test_accuracy']]
    right_scores = [right_grid_results['best_score'], right_grid_results['test_accuracy']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, left_scores, width, label='좌수', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, right_scores, width, label='우수', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('평가 지표')
    ax1.set_ylabel('점수')
    ax1.set_title('성능 비교')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    for i, (left, right) in enumerate(zip(left_scores, right_scores)):
        ax1.text(i - width/2, left + 0.01, f'{left:.3f}', ha='center', va='bottom')
        ax1.text(i + width/2, right + 0.01, f'{right:.3f}', ha='center', va='bottom')
    
    left_kernels = list(left_grid_results['kernel_performance'].keys())
    left_f1_scores = [left_grid_results['kernel_performance'][k]['f1_score'] for k in left_kernels]
    
    ax2.bar(left_kernels, left_f1_scores, alpha=0.8, color='skyblue')
    ax2.set_xlabel('커널')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('좌수 커널별 성능')
    ax2.grid(True, alpha=0.3)
    
    for i, score in enumerate(left_f1_scores):
        ax2.text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom')
    
    right_kernels = list(right_grid_results['kernel_performance'].keys())
    right_f1_scores = [right_grid_results['kernel_performance'][k]['f1_score'] for k in right_kernels]
    
    ax3.bar(right_kernels, right_f1_scores, alpha=0.8, color='lightcoral')
    ax3.set_xlabel('커널')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('우수 커널별 성능')
    ax3.grid(True, alpha=0.3)
    
    for i, score in enumerate(right_f1_scores):
        ax3.text(i, score + 0.005, f'{score:.3f}', ha='center', va='bottom')
    
    feature_data = ['좌수', '우수']
    feature_counts = [len(left_features), len(right_features)]
    
    bars = ax4.bar(feature_data, feature_counts, alpha=0.8, color=['skyblue', 'lightcoral'])
    ax4.set_xlabel('데이터 유형')
    ax4.set_ylabel('특성 개수')
    ax4.set_title('선택된 특성 개수')
    ax4.grid(True, alpha=0.3)
    
    for bar, count in zip(bars, feature_counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('final_left_right_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

create_final_comparison()

if left_grid_results['best_score'] > right_grid_results['best_score']:
    winner = "좌수"
    winner_score = left_grid_results['best_score']
    winner_features = left_features
    winner_params = left_grid_results['best_params']
else:
    winner = "우수"
    winner_score = right_grid_results['best_score']
    winner_features = right_features
    winner_params = right_grid_results['best_params']

print(f"\n🏆 최종 결과")
print("-" * 50)
print(f"승자: {winner} 데이터")
print(f"최고 성능: F1-Score {winner_score:.4f}")
print(f"특성 개수: {len(winner_features)}개")
print(f"최적 파라미터: {winner_params}")

print(f"\n💾 복사용 최종 설정")
print("-" * 50)
print(f"# 최고 성능 설정")
print(f"winner_data = '{winner}'")
print(f"best_features = {winner_features}")
print(f"best_params = {winner_params}")
print(f"best_score = {winner_score:.4f}")

print(f"\n# 좌수 설정")
print(f"left_features = {left_features}")
print(f"left_params = {left_grid_results['best_params']}")

print(f"\n# 우수 설정")
print(f"right_features = {right_features}")
print(f"right_params = {right_grid_results['best_params']}")

print(f"\n✅ 모든 분석 완료!")
print(f"생성된 파일: 좌수_gridsearch_heatmap.png, 우수_gridsearch_heatmap.png, final_left_right_comparison.png")