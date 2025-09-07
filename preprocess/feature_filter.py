import pandas as pd
from scipy.stats import ttest_ind

try:
    # 1. 두 개의 CSV 파일 불러오기
    # 파일 경로를 'data/' 폴더 안으로 수정했습니다.
    features_df = pd.read_csv('data/all_features_left.csv')
    labels_df = pd.read_csv('data/labels_all_left.csv')

    # 2. 데이터 합치기
    df = pd.concat([features_df, labels_df], axis=1)

    # 3. 그룹 분리
    target_column = labels_df.columns[0]
    cold_label = 1
    hot_label = 2
    
    group_cold = df[df[target_column] == cold_label]
    group_hot = df[df[target_column] == hot_label]

    print(f"--- 데이터 그룹 정보 ---")
    print(f"Target 컬럼: '{target_column}'")
    print(f"'한' 그룹 (라벨 {cold_label}): {len(group_cold)} 명")
    print(f"'열' 그룹 (라벨 {hot_label}): {len(group_hot)} 명\n")

    # 4. t-검정 수행
    features_to_test = features_df.columns.tolist()
    results = []
    for feature in features_to_test:
        cold_data = group_cold[feature].dropna()
        hot_data = group_hot[feature].dropna()
        
        if len(cold_data) > 1 and len(hot_data) > 1:
            t_statistic, p_value = ttest_ind(cold_data, hot_data)
            results.append({
                'Feature': feature,
                'T-statistic': t_statistic,
                'P-value': p_value
            })

    # 5. 결과 확인 및 출력
    if results:
        results_df = pd.DataFrame(results)
        results_df_sorted = results_df.sort_values(by='P-value', ascending=True)

        # --- 기존 코드: 유의미한 특성 (P < 0.05)만 출력 ---
        significant_features_df = results_df_sorted[results_df_sorted['P-value'] < 0.05]
        print("--- 통계적으로 유의미한 특성 (P < 0.05) ---")
        print(significant_features_df)
        
        significant_feature_list = significant_features_df['Feature'].tolist()
        print("\n--- P<0.05 기준 후보 특성 목록 ---")
        print(significant_feature_list)

        # --- ✨ 새로 추가된 부분: 상위 20개 특성 전체 출력 ---
        print("\n\n--- P-value 기준 상위 20개 특성 ---")
        top_20_features = results_df_sorted.head(20)
        # to_string()을 사용해 모든 행이 잘리지 않고 출력되도록 함
        print(top_20_features.to_string())
        
    else:
        print("분석할 특성이 없거나 데이터가 부족합니다.")

except FileNotFoundError:
    print("오류: 'data/' 폴더 안에 'all_features_right.csv' 또는 'labels_all_right.csv' 파일이 있는지 확인해주세요.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")