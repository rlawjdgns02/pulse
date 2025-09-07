import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

try:
    # --- 1. 데이터 로드 및 준비 ---
    # 사용자가 제공한 파일 이름을 정확히 사용
    features_df = pd.read_csv('data/all_features_left.csv')
    labels_df = pd.read_csv('data/labels_all_left.csv')
    
    # 라벨을 0(한), 1(열)로 변환 (AUC 계산을 위해 필요)
    target_column = labels_df.columns[0]
    # '한'==1 -> 0, '열'==2 -> 1
    y = labels_df[target_column].apply(lambda x: 0 if x == 1 else 1)

    features_to_test = features_df.columns.tolist()
    
    print(f"--- 총 {len(features_to_test)}개 특성에 대한 ROC-AUC 분석을 시작합니다. ---")

    # --- 2. 모든 특성에 대해 AUC 점수 계산 ---
    results = []
    
    # tqdm을 사용하여 진행률 표시
    for feature in tqdm(features_to_test, desc="AUC 계산 진행 중"):
        
        # 현재 특성과 라벨을 합쳐 결측값 제거
        temp_df = pd.concat([features_df[[feature]], y], axis=1).dropna()
        
        X_feature = temp_df[[feature]]
        y_true = temp_df[target_column]
        
        # 데이터에 두 클래스가 모두 존재하는지 확인
        if len(np.unique(y_true)) < 2:
            continue
            
        # AUC 점수 계산
        auc_score = roc_auc_score(y_true, X_feature)
        
        results.append({
            'Feature': feature,
            'AUC_Score': auc_score
        })

    # --- 3. 최종 결과 분석 및 출력 ---
    if results:
        results_df = pd.DataFrame(results)
        
        # AUC 점수가 0.5에서 얼마나 멀리 떨어져 있는지를 '분리력'으로 계산
        results_df['Separation_Power'] = np.abs(results_df['AUC_Score'] - 0.5) * 2
        
        # 분리력을 기준으로 내림차순 정렬
        results_df_sorted = results_df.sort_values(by='Separation_Power', ascending=False)
        
        print("\n\n--- ROC-AUC 기반 특성 중요도 순위 ---")
        with pd.option_context('display.max_rows', None):
            print(results_df_sorted)

        # 분리력이 높은 (AUC > 0.7 또는 < 0.3) 특성들을 유의미하다고 간주
        significant_features = results_df_sorted[results_df_sorted['Separation_Power'] > 0.2]
        
        significant_feature_list = significant_features['Feature'].tolist()
        print("\n\n--- AUC 기반 최종 후보 특성 목록 (Separation Power > 0.4) ---")
        print(f"총 {len(significant_feature_list)}개의 유의미한 특성을 찾았습니다.")
        print(significant_feature_list)

    else:
        print("\n분석을 수행할 데이터가 없습니다.")

except FileNotFoundError:
    print("오류: 'all_features_left.csv' 또는 'labels_all_left.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")