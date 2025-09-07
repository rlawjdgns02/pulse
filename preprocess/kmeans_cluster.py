import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans             # K-평균 클러스터링을 위해 추가
from sklearn.preprocessing import MinMaxScaler # 스케일링을 위해 추가

try:
    # --- 1. 데이터 로드 ---
    # 사용자가 지정한 파일 이름을 정확히 사용합니다.
    features_df = pd.read_csv('features_left.csv')
    labels_df = pd.read_csv('labels_left.csv')

    # 두 데이터를 하나로 합치기
    df = pd.concat([features_df, labels_df], axis=1)
    
    print("--- 데이터 로드 및 병합 완료 ---")
    print(f"특성 파일: 'features_left.csv' (크기: {features_df.shape})")
    print(f"라벨 파일: 'labels_left.csv' (크기: {labels_df.shape})")

    
    # ==================================================================
    # [기존 코드] 실제 '한/열' 라벨 기준 시각화
    # ==================================================================
    
    print("\n--- 1. 실제 '한/열' 라벨 기준 시각화 생성 중 ---")
    
    # P-value가 가장 낮았던 두 특성을 x, y축으로 사용
    x_axis_feature = 'Cardiac Output_left'
    y_axis_feature = 'Pulse Pressure_left'
    
    # 라벨 컬럼명 (labels_df의 첫 번째 컬럼)
    label_column = labels_df.columns[0]
    
    plt.figure(figsize=(12, 8))
    
    sns.scatterplot(
        data=df,
        x=x_axis_feature,
        y=y_axis_feature,
        hue=label_column,
        palette='coolwarm',
        s=100,
        alpha=0.8,
        style=label_column,
        markers=['o', 's']
    )
    
    plt.title("Visualization by Actual 'Han/Yeol' Labels", fontsize=16)
    plt.xlabel(f'Feature: {x_axis_feature}', fontsize=12)
    plt.ylabel(f'Feature: {y_axis_feature}', fontsize=12)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = ['Han (Cold)', 'Yeol (Heat)'] 
    plt.legend(handles=handles[1:3], labels=new_labels, title='Diagnosis')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_filename_1 = 'han_yeol_visualization.png'
    plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
    print(f"그래프가 '{output_filename_1}' 파일로 저장되었습니다.")
    
    plt.show()

    
    # ==================================================================
    # [추가된 부분] K-평균 군집 분석 및 시각화
    # ==================================================================
    
    print("\n--- 2. K-평균 군집 분석 및 시각화 생성 중 ---")
    
    # 스케일링
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features_df)
    
    # K-평균 군집 분석 실행 (k=3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(features_scaled)
    
    # 원본 데이터프레임에 클러스터 결과 추가
    df['cluster'] = kmeans.labels_
    
    print("클러스터별 샘플 수:")
    print(df['cluster'].value_counts())
    
    plt.figure(figsize=(12, 8))
    
    # hue 옵션에 K-평균이 만든 'cluster' 컬럼을 지정
    sns.scatterplot(
        data=df,
        x=x_axis_feature,
        y=y_axis_feature,
        hue='cluster',
        palette='viridis',
        s=100,
        alpha=0.8
    )

    plt.title('K-Means Clustering Results (k=3)', fontsize=16)
    plt.xlabel(f'Feature: {x_axis_feature}', fontsize=12)
    plt.ylabel(f'Feature: {y_axis_feature}', fontsize=12)
    plt.legend(title='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_filename_2 = 'kmeans_clustering_visualization.png'
    plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
    print(f"그래프가 '{output_filename_2}' 파일로 저장되었습니다.")
    
    plt.show()


except FileNotFoundError:
    print("오류: 'features_left.csv' 또는 'labels_left.csv' 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")