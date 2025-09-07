import pandas as pd

# 파일 경로
original_csv_file_left = 'data/all_features_left.csv'
original_csv_file_right = 'data/all_features_right.csv'
labels_file_left = 'data/labels_all_left.csv'
labels_file_right = 'data/labels_all_right.csv'

# 좌수 (Left) - 확장된 맥진 분석
features_left_cols = ['Cardiac Output_left', 'Pulse Pressure_left', 'ECR_Left', 'HRV_LH_ratio_Left', 'W_R_Left']
features_right_cols = ['Pulse Rate_Right', 'Systolic Blood Pressure_Right', 'Diastolic Blood Pressure_Right',
                          'Mean Blood Pressure_Right', 'Pulse Pressure_Right', 'Strength_Right', 'Depth_Right', 
                          'HRV_LF_Right', 'HRV_HF_Right', 'HRV_LH_ratio_Right', 'ESI_Right', 'RAI_Right',
                          'Gradient_I1_Right', 'Gradient_D1_Right', 'H4/H1_R_Right', 'W_R_Right']

def process_and_save_data(features_df, labels_df, feature_list, side_name):
    print(f"--- {side_name.capitalize()} 손 데이터 처리 시작 ---")
    
    try:
        # 1. 특성 선택
        df_features = features_df[feature_list]
        
        # 2. 특성과 라벨 결합
        df_combined = pd.concat([df_features, labels_df], axis=1)
        rows_before = len(df_combined)
        print(f"처리 전 행 개수: {rows_before}")
        
        # 3. 결측값 제거
        df_cleaned = df_combined.dropna(axis=0)
        rows_after = len(df_cleaned)
        print(f"처리 후 행 개수: {rows_after} ({rows_before - rows_after}개 행 제거)")
        
        # 4. 특성과 라벨 분리
        final_features = df_cleaned[feature_list]
        final_labels = df_cleaned.iloc[:, -1:]  # 마지막 컬럼 (라벨)
        
        # 5. 저장
        final_features.to_csv(f'features_{side_name}.csv', index=False)
        final_labels.to_csv(f'labels_{side_name}.csv', index=False)
        
        print(f"'features_{side_name}.csv'와 'labels_{side_name}.csv' 파일 생성 완료.\n")
        
    except Exception as e:
        print(f"오류: {e}")

if __name__ == "__main__":
    # 파일 로드
    main_df_left = pd.read_csv(original_csv_file_left)
    main_df_right = pd.read_csv(original_csv_file_right)
    labels_left = pd.read_csv(labels_file_left)
    labels_right = pd.read_csv(labels_file_right)
    
    # 처리
    # process_and_save_data(main_df_right, labels_right, features_right_cols, 'right')
    process_and_save_data(main_df_left, labels_left, features_left_cols, 'left')