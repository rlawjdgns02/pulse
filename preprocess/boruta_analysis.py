import pandas as pd
import numpy as np
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 로드
df = pd.read_csv('data/all_features_left.csv')
print(f"데이터 형태: {df.shape}")
print(f"컬럼명: {list(df.columns)}")

# 2. 타겟 변수 설정 (여기서 실제 타겟 컬럼명으로 수정 필요)
# 예시: 'target' 또는 '한열분류' 등의 컬럼명
# target_column = 'your_target_column'  # 실제 타겟 컬럼명으로 변경
# X = df.drop(target_column, axis=1)
# y = df[target_column]

# 임시로 타겟이 없다면 더미 타겟 생성 (실제로는 실제 타겟 사용)
print("⚠️ 타겟 변수를 찾을 수 없어 더미 타겟 생성 중...")
X = df
y = np.random.choice([0, 1], size=len(df))  # 실제 타겟으로 교체 필요
print("실제 사용 시에는 위 2줄을 삭제하고 실제 타겟 변수를 사용하세요!")

print(f"특성 수: {X.shape[1]}")
print(f"샘플 수: {X.shape[0]}")
print(f"타겟 분포: {np.bincount(y)}")

# 3. Boruta 특성 선택
print("\n🔍 Boruta 특성 선택 시작...")
print("⏰ 약 2-5분 소요될 예정입니다...")

# RandomForest 모델 설정 (작은 데이터셋에 맞게 조정)
rf = RandomForestClassifier(
    n_estimators=100,           # 트리 개수
    max_depth=5,               # 깊이 제한 (과적합 방지)
    min_samples_split=5,       # 분할 최소 샘플
    min_samples_leaf=2,        # 리프 최소 샘플
    random_state=42,
    n_jobs=-1                  # 병렬 처리
)

# Boruta 설정
boruta = BorutaPy(
    rf,
    n_estimators='auto',       # 자동으로 반복 횟수 결정
    max_iter=100,             # 최대 반복 횟수
    alpha=0.05,               # 유의 수준
    random_state=42,
    verbose=2                 # 진행 상황 출력
)

# Boruta 실행
boruta.fit(X.values, y)

# 4. 결과 분석
print("\n📊 Boruta 결과 분석")
print(f"확정된 중요 특성: {boruta.n_features_}")
print(f"불확실한 특성: {sum(boruta.support_weak_)}")
print(f"거부된 특성: {sum(~boruta.support_)}")

# 특성별 상태 출력
feature_status = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': boruta.ranking_,
    'Confirmed': boruta.support_,
    'Tentative': boruta.support_weak_
})

print("\n🔝 확정된 중요 특성들:")
confirmed_features = feature_status[feature_status['Confirmed']]['Feature'].tolist()
for feature in confirmed_features:
    print(f"  ✅ {feature}")

print(f"\n📈 불확실한 특성들 ({sum(boruta.support_weak_)}개):")
tentative_features = feature_status[feature_status['Tentative']]['Feature'].tolist()
for feature in tentative_features[:10]:  # 처음 10개만 출력
    print(f"  ❓ {feature}")

# 5. 선택된 특성으로 모델 성능 검증
if len(confirmed_features) > 0:
    print(f"\n🎯 선택된 {len(confirmed_features)}개 특성으로 모델 성능 검증")
    
    X_selected = X[confirmed_features]
    
    # 교차 검증 (작은 데이터셋에 적합하게 설정)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_selected, y, cv=cv, scoring='accuracy')
    
    print(f"교차 검증 정확도: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"각 fold 점수: {cv_scores}")
    
    # 전체 데이터로 모델 훈련 (참고용)
    rf.fit(X_selected, y)
    feature_importance = pd.DataFrame({
        'Feature': confirmed_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 선택된 특성들의 중요도:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

else:
    print("⚠️ 확정된 특성이 없습니다. alpha 값을 높이거나 다른 방법을 시도해보세요.")

# 6. 선택된 특성으로 최종 데이터셋 생성
if len(confirmed_features) > 0:
    # 선택된 특성 + 레이블로 최종 데이터셋 생성
    final_dataset = X[confirmed_features].copy()
    final_dataset['한/열'] = y + 1  # 다시 1,2로 복원
    final_dataset.to_csv('selected_features_with_labels.csv', index=False)
    print(f"\n💾 선택된 특성 + 레이블을 'selected_features_with_labels.csv'로 저장했습니다.")
    
    # 선택된 특성만 저장
    selected_data = X[confirmed_features]
    selected_data.to_csv('selected_features_boruta.csv', index=False)
    print(f"💾 선택된 특성만 'selected_features_boruta.csv'로 저장했습니다.")
    
    # 한의학적 해석을 위한 특성 카테고리 분석
    print(f"\n🏥 선택된 특성의 한의학적 카테고리 분석:")
    categories = {
        "기본 생체신호": [],
        "맥파 고조파": [],
        "맥파 시간": [],
        "변동성": [],
        "기타": []
    }
    
    for feature in confirmed_features:
        if any(x in feature for x in ['Blood Pressure', 'Pulse Rate', 'Cardiac']):
            categories["기본 생체신호"].append(feature)
        elif 'H' in feature and any(str(i) in feature for i in range(1,6)):
            categories["맥파 고조파"].append(feature)
        elif 'T' in feature and any(str(i) in feature for i in range(1,6)):
            categories["맥파 시간"].append(feature)
        elif any(x in feature for x in ['STD', 'Var', 'HRV']):
            categories["변동성"].append(feature)
        else:
            categories["기타"].append(feature)
    
    for category, features in categories.items():
        if features:
            print(f"  {category}: {len(features)}개")
            for f in features:
                print(f"    - {f}")

else:
    print("⚠️ 확정된 특성이 없습니다. 파라미터를 조정해보세요.")
    print("💡 권장 조치:")
    print("  - alpha 값을 0.1로 증가")
    print("  - max_iter를 150으로 증가")
    print("  - 또는 다른 특성 선택 방법 시도")

# 특성 선택 결과 저장
feature_status.to_csv('boruta_feature_analysis.csv', index=False)
print("📊 특성 분석 결과를 'boruta_feature_analysis.csv'로 저장했습니다.")

print("\n" + "="*50)
print("🎉 Boruta 특성 선택 완료!")
print("="*50)