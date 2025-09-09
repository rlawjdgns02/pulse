import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold, GridSearchCV, train_test_split, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

# 영어 폰트 설정 (한글 폰트 문제 방지)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("IMPROVED Feature Selection & Model Optimization for Small Dataset")
print("=" * 80)
print("Key Improvements:")
print("- Fixed data leakage with proper pipelines")
print("- Simplified 2-step feature selection") 
print("- Added multiple comparison correction")
print("- Implemented dimension reduction options")
print("- Proper cross-validation strategy")
print("=" * 80)

# 1. 데이터 로드 및 기본 분석
print("\n1. Data Loading & Basic Analysis")
print("-" * 50)

try:
    left_features = pd.read_csv('data/all_features_left.csv')
    left_labels = pd.read_csv('data/labels_all_left.csv')
    right_features = pd.read_csv('data/all_features_right.csv')
    right_labels = pd.read_csv('data/labels_all_right.csv')
    
    print(f"✓ Data loaded successfully")
    print(f"  Left hand: {left_features.shape[0]} samples, {left_features.shape[1]} features")
    print(f"  Right hand: {right_features.shape[0]} samples, {right_features.shape[1]} features")
    
    # 차원의 저주 경고
    if left_features.shape[1] > left_features.shape[0]:
        print(f"⚠️  WARNING: Features ({left_features.shape[1]}) > Samples ({left_features.shape[0]}) - Curse of dimensionality!")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# 2. 통계적 특성 분석 함수 (다중비교 보정 포함)
def statistical_feature_analysis(features, labels, name):
    """통계적 특성 분석 with 다중비교 보정"""
    print(f"\n{'='*60}")
    print(f"Statistical Analysis: {name}")
    print(f"{'='*60}")
    
    X, y = features.copy(), labels.iloc[:, 0].copy()
    y_binary = y - 1  # 1,2 → 0,1 변환
    
    print(f"Original data: {X.shape}")
    print(f"Class distribution: Cold(0)={np.sum(y_binary==0)}, Hot(1)={np.sum(y_binary==1)}")
    
    # t-test for all features
    cold_group = X[y_binary == 0]
    hot_group = X[y_binary == 1]
    
    t_stats = []
    p_values = []
    feature_names = []
    
    for col in X.columns:
        if X[col].std() > 1e-10:  # 분산이 있는 특성만
            cold_data = cold_group[col].dropna()
            hot_data = hot_group[col].dropna()
            
            if len(cold_data) > 5 and len(hot_data) > 5:
                t_stat, p_val = ttest_ind(cold_data, hot_data)
                t_stats.append(abs(t_stat))
                p_values.append(p_val)
                feature_names.append(col)
    
    # 다중비교 보정
    if len(p_values) > 0:
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, method='fdr_bh', alpha=0.1
        )
        
        # 결과 정리
        results_df = pd.DataFrame({
            'Feature': feature_names,
            'T_statistic': t_stats,
            'P_value': p_values,
            'P_corrected_FDR': p_corrected,
            'Significant_FDR': rejected
        }).sort_values('P_corrected_FDR')
        
        print(f"\n📊 Statistical Test Results:")
        print(f"  Total features tested: {len(feature_names)}")
        print(f"  Significant (raw p<0.05): {np.sum(np.array(p_values) < 0.05)}")
        print(f"  Significant (FDR corrected): {np.sum(rejected)}")
        
        if np.sum(rejected) > 0:
            print(f"\n🎯 Top 10 significant features (FDR corrected):")
            significant_features = results_df[results_df['Significant_FDR']].head(10)
            print(significant_features[['Feature', 'T_statistic', 'P_corrected_FDR']].to_string(index=False))
        
        return results_df
    else:
        print("❌ No features passed basic variance check")
        return pd.DataFrame()

# 3. 개선된 특성 선택 파이프라인 함수
def create_feature_selection_pipeline(max_features=15):
    """데이터 누출 방지를 위한 파이프라인 생성"""
    
    # 옵션 1: 단순한 univariate 선택
    pipeline_simple = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('selector', SelectKBest(f_classif, k=min(max_features, 20))),
        ('scaler', StandardScaler()),
        ('classifier', SVC(random_state=42))
    ])
    
    # 옵션 2: PCA 차원 축소
    pipeline_pca = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=min(max_features, 20))),
        ('classifier', SVC(random_state=42))
    ])
    
    # 옵션 3: LDA 차원 축소 (감독 학습)
    pipeline_lda = Pipeline([
        ('variance', VarianceThreshold(threshold=0.01)),
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis()),
        ('classifier', SVC(random_state=42))
    ])
    
    return {
        'Simple_SelectKBest': pipeline_simple,
        'PCA_Reduction': pipeline_pca,
        'LDA_Reduction': pipeline_lda
    }

# 4. 모델 비교 및 최적화
def compare_pipelines_and_optimize(X, y, name, max_features=15):
    """파이프라인 비교 및 하이퍼파라미터 최적화"""
    print(f"\n{'='*60}")
    print(f"Pipeline Comparison & Optimization: {name}")
    print(f"{'='*60}")
    
    # 안정적인 교차 검증 (반복)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    pipelines = create_feature_selection_pipeline(max_features)
    results = {}
    
    print(f"\n🔄 Testing {len(pipelines)} different approaches...")
    
    # 각 파이프라인 성능 비교
    for pipe_name, pipeline in pipelines.items():
        try:
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
            results[pipe_name] = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores,
                'pipeline': pipeline
            }
            print(f"  {pipe_name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
        except Exception as e:
            print(f"  {pipe_name:20s}: Failed ({str(e)[:30]}...)")
            results[pipe_name] = {'mean_score': 0, 'std_score': 0}
    
    # 최고 성능 파이프라인 선택
    best_pipe_name = max(results.keys(), key=lambda x: results[x]['mean_score'])
    best_pipeline = results[best_pipe_name]['pipeline']
    
    print(f"\n🏆 Best approach: {best_pipe_name}")
    print(f"   Score: {results[best_pipe_name]['mean_score']:.4f} ± {results[best_pipe_name]['std_score']:.4f}")
    
    # 최고 파이프라인에 대해 하이퍼파라미터 튜닝
    print(f"\n🔧 Hyperparameter tuning for best pipeline...")
    
    # SVM 파라미터 그리드 (작은 데이터셋에 맞게 조정)
    param_grid = [
        {
            f'classifier__kernel': ['linear'],
            f'classifier__C': [0.01, 0.1, 1, 10, 100]
        },
        {
            f'classifier__kernel': ['rbf'],
            f'classifier__C': [0.01, 0.1, 1, 10, 100],
            f'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
    ]
    
    # SelectKBest인 경우 k 파라미터도 튜닝
    if 'selector' in best_pipeline.named_steps:
        n_features_available = X.shape[1] - 1  # variance filter 고려
        k_options = [min(k, n_features_available) for k in [5, 10, 15, 20] if k <= n_features_available]
        if k_options:
            for grid in param_grid:
                grid['selector__k'] = k_options
    
    try:
        grid_search = GridSearchCV(
            best_pipeline, param_grid, cv=cv, scoring='f1_macro',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        print(f"✓ Best parameters: {grid_search.best_params_}")
        print(f"✓ Best CV score: {grid_search.best_score_:.4f}")
        
        return {
            'name': name,
            'best_pipeline_name': best_pipe_name,
            'best_estimator': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'pipeline_results': results,
            'grid_results': grid_search.cv_results_
        }
        
    except Exception as e:
        print(f"❌ Grid search failed: {e}")
        return {
            'name': name,
            'best_pipeline_name': best_pipe_name,
            'best_estimator': best_pipeline,
            'best_params': {},
            'best_score': results[best_pipe_name]['mean_score'],
            'pipeline_results': results
        }

# 5. 결과 시각화 함수
def create_comprehensive_visualization(left_results, right_results, left_stats, right_stats):
    """종합적인 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Feature Selection & Model Analysis', fontsize=16, fontweight='bold')
    
    # 1. 파이프라인 성능 비교 - 좌수
    pipe_names = list(left_results['pipeline_results'].keys())
    left_scores = [left_results['pipeline_results'][name]['mean_score'] for name in pipe_names]
    left_stds = [left_results['pipeline_results'][name]['std_score'] for name in pipe_names]
    
    axes[0,0].bar(range(len(pipe_names)), left_scores, yerr=left_stds, 
                 alpha=0.7, color='skyblue', capsize=5)
    axes[0,0].set_xticks(range(len(pipe_names)))
    axes[0,0].set_xticklabels(pipe_names, rotation=45, ha='right')
    axes[0,0].set_ylabel('F1-Score')
    axes[0,0].set_title('Left Hand: Pipeline Comparison')
    axes[0,0].grid(True, alpha=0.3)
    
    # 값 표시
    for i, (score, std) in enumerate(zip(left_scores, left_stds)):
        axes[0,0].text(i, score + std + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    # 2. 파이프라인 성능 비교 - 우수
    right_scores = [right_results['pipeline_results'][name]['mean_score'] for name in pipe_names]
    right_stds = [right_results['pipeline_results'][name]['std_score'] for name in pipe_names]
    
    axes[0,1].bar(range(len(pipe_names)), right_scores, yerr=right_stds, 
                 alpha=0.7, color='lightcoral', capsize=5)
    axes[0,1].set_xticks(range(len(pipe_names)))
    axes[0,1].set_xticklabels(pipe_names, rotation=45, ha='right')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].set_title('Right Hand: Pipeline Comparison')
    axes[0,1].grid(True, alpha=0.3)
    
    for i, (score, std) in enumerate(zip(right_scores, right_stds)):
        axes[0,1].text(i, score + std + 0.01, f'{score:.3f}', ha='center', va='bottom')
    
    # 3. 좌수 vs 우수 최종 성능 비교
    final_scores = [left_results['best_score'], right_results['best_score']]
    methods = [left_results['best_pipeline_name'], right_results['best_pipeline_name']]
    
    bars = axes[0,2].bar(['Left Hand', 'Right Hand'], final_scores, 
                        alpha=0.7, color=['skyblue', 'lightcoral'])
    axes[0,2].set_ylabel('Best F1-Score')
    axes[0,2].set_title('Final Performance Comparison')
    axes[0,2].grid(True, alpha=0.3)
    
    for bar, score, method in zip(bars, final_scores, methods):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}\n({method})', ha='center', va='bottom', fontsize=10)
    
    # 4. 통계적 유의성 분석 - 좌수
    if not left_stats.empty:
        top_features_left = left_stats.head(15)
        axes[1,0].barh(range(len(top_features_left)), -np.log10(top_features_left['P_corrected_FDR']))
        axes[1,0].set_yticks(range(len(top_features_left)))
        axes[1,0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                                  for f in top_features_left['Feature']], fontsize=8)
        axes[1,0].set_xlabel('-log10(P-value corrected)')
        axes[1,0].set_title('Left: Top Significant Features')
        axes[1,0].axvline(x=-np.log10(0.1), color='red', linestyle='--', alpha=0.7, label='FDR=0.1')
        axes[1,0].legend()
    else:
        axes[1,0].text(0.5, 0.5, 'No significant features', ha='center', va='center',
                      transform=axes[1,0].transAxes)
        axes[1,0].set_title('Left: No Significant Features')
    
    # 5. 통계적 유의성 분석 - 우수
    if not right_stats.empty:
        top_features_right = right_stats.head(15)
        axes[1,1].barh(range(len(top_features_right)), -np.log10(top_features_right['P_corrected_FDR']))
        axes[1,1].set_yticks(range(len(top_features_right)))
        axes[1,1].set_yticklabels([f[:20] + '...' if len(f) > 20 else f 
                                  for f in top_features_right['Feature']], fontsize=8)
        axes[1,1].set_xlabel('-log10(P-value corrected)')
        axes[1,1].set_title('Right: Top Significant Features')
        axes[1,1].axvline(x=-np.log10(0.1), color='red', linestyle='--', alpha=0.7, label='FDR=0.1')
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0.5, 'No significant features', ha='center', va='center',
                      transform=axes[1,1].transAxes)
        axes[1,1].set_title('Right: No Significant Features')
    
    # 6. 데이터 품질 요약
    axes[1,2].axis('off')
    
    # 데이터 요약 텍스트
    summary_text = f"""Data Quality Summary:

Left Hand:
• Samples: {left_features.shape[0]}
• Features: {left_features.shape[1]}
• Feature/Sample Ratio: {left_features.shape[1]/left_features.shape[0]:.2f}
• Best Method: {left_results['best_pipeline_name']}
• Best Score: {left_results['best_score']:.4f}

Right Hand:
• Samples: {right_features.shape[0]}  
• Features: {right_features.shape[1]}
• Feature/Sample Ratio: {right_features.shape[1]/right_features.shape[0]:.2f}
• Best Method: {right_results['best_pipeline_name']}
• Best Score: {right_results['best_score']:.4f}

Recommendations:
• {'⚠️ High dimensional data' if left_features.shape[1] > left_features.shape[0] else '✓ Reasonable dimensions'}
• {'⚠️ Small sample size' if left_features.shape[0] < 200 else '✓ Adequate sample size'}
• Use regularization/dimension reduction
• Collect more data if possible"""

    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('improved_eda_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. 메인 실행부
print("\n" + "="*80)
print("MAIN ANALYSIS")
print("="*80)

# 통계적 분석
left_stats = statistical_feature_analysis(left_features, left_labels, "Left Hand")
right_stats = statistical_feature_analysis(right_features, right_labels, "Right Hand")

# 모델 비교 및 최적화 (작은 데이터셋에 맞게 특성 수 제한)
max_features = min(15, left_features.shape[0] // 6)  # 샘플 수의 1/6 이하

left_y_binary = left_labels.iloc[:, 0] - 1
right_y_binary = right_labels.iloc[:, 0] - 1

left_results = compare_pipelines_and_optimize(left_features, left_y_binary, "Left Hand", max_features)
right_results = compare_pipelines_and_optimize(right_features, right_y_binary, "Right Hand", max_features)

# 종합 시각화
create_comprehensive_visualization(left_results, right_results, left_stats, right_stats)

# 7. 최종 결과 및 권장사항
print("\n" + "="*80)
print("FINAL RESULTS & RECOMMENDATIONS")
print("="*80)

# 승자 결정
if left_results['best_score'] > right_results['best_score']:
    winner = "Left Hand"
    winner_results = left_results
    winner_stats = left_stats
else:
    winner = "Right Hand"
    winner_results = right_results
    winner_stats = right_stats

print(f"\n🏆 WINNER: {winner}")
print(f"   Method: {winner_results['best_pipeline_name']}")
print(f"   Best Score: {winner_results['best_score']:.4f}")
print(f"   Parameters: {winner_results['best_params']}")

print(f"\n📊 COMPARISON TABLE:")
print("-" * 70)
comparison_df = pd.DataFrame({
    'Metric': ['Best Method', 'F1-Score', 'Kernel', 'C Parameter', 'Other Params'],
    'Left Hand': [
        left_results['best_pipeline_name'],
        f"{left_results['best_score']:.4f}",
        left_results['best_params'].get('classifier__kernel', 'N/A'),
        left_results['best_params'].get('classifier__C', 'N/A'),
        {k:v for k,v in left_results['best_params'].items() 
         if not k.startswith('classifier__kernel') and not k.startswith('classifier__C')}
    ],
    'Right Hand': [
        right_results['best_pipeline_name'],
        f"{right_results['best_score']:.4f}",
        right_results['best_params'].get('classifier__kernel', 'N/A'),
        right_results['best_params'].get('classifier__C', 'N/A'),
        {k:v for k,v in right_results['best_params'].items() 
         if not k.startswith('classifier__kernel') and not k.startswith('classifier__C')}
    ]
})
print(comparison_df.to_string(index=False))

print(f"\n💡 KEY IMPROVEMENTS APPLIED:")
print("-" * 50)
print("✓ Fixed data leakage with proper Pipeline")
print("✓ Simplified feature selection (2-step process)")
print("✓ Added multiple comparison correction (FDR)")
print("✓ Implemented dimension reduction techniques")
print("✓ Used repeated cross-validation for stability")
print("✓ Limited feature count for small dataset")
print("✓ Added comprehensive visualization")

print(f"\n🎯 NEXT STEPS RECOMMENDATIONS:")
print("-" * 50)
print("1. 📈 Data Collection: Aim for 300-500 samples minimum")
print("2. 🔍 Domain Knowledge: Consult medical experts for feature engineering")
print("3. 🧪 Advanced Methods: Try XGBoost, Neural Networks with regularization")
print("4. 📊 Feature Engineering: Create interaction features, polynomial terms")
print("5. 🔄 Ensemble Methods: Combine multiple models for robustness")

print(f"\n💾 COPY-READY CONFIGURATION:")
print("-" * 50)
print(f"# Winner Configuration")
print(f"winner_data = '{winner}'")
print(f"best_method = '{winner_results['best_pipeline_name']}'")
print(f"best_params = {winner_results['best_params']}")
print(f"best_score = {winner_results['best_score']:.4f}")

print(f"\n✅ Analysis Complete!")
print(f"Generated file: improved_eda_analysis.png")
print("="*80)