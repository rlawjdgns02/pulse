# 맥진 데이터를 이용한 한/열 체질 분류 프로젝트

## 📋 프로젝트 개요
맥진기를 통해 수집한 맥파 데이터를 활용하여 한의학의 한/열 체질 분류가 가능한지 확인하는 머신러닝 프로젝트입니다.

## 🔍 데이터 정보
- **원시 데이터**: 맥진기에서 수집된 맥파 raw 데이터
- **전처리 데이터**: 업체에서 처리하여 제공된 198개 특성 (좌수 99개, 우수 99개)
- **샘플 수**: 약 90-95명 (결측값으로 인한 변동)
- **클래스 분포**: 한(1): 47명, 열(2): 42명 (비교적 균형잡힌 분포)

## 🎯 프로젝트 목표
1. **1차 목표**: 한/열 체질 분류 가능성 확인
2. **2차 목표**: 분류 정확도 90% 이상 달성
3. **향후 계획**: 허/실 체질 분류 확장

## 🔧 분석 파이프라인

### 1. 특성 선택 (EDA)
- **원본**: 198개 특성 (좌수/우수 각 99개)
- **축소**: 8-10개 핵심 특성으로 차원 축소
- **방법**: 상관관계 분석, 타겟 연관성 분석, 통계적 유의성 검정

### 2. 선택된 주요 특성
**우수 (10개):**
- Systemic Vascular Resistance Index, H4, Systolic Blood Pressure
- W_Area_R, ECO(%), stdRespi, Wfb_ratio_R
- Gradient_D1, MeanResp, Cardiac Output

**좌수 (8개):**
- H4, ECO(%), Cardiac Output, Ap, H2_f_R
- STD_PR_R, Pulse Rate, Var_period

### 3. 모델링
**현재 시도한 모델:**
- **SVM**: 68-72% 정확도 (하이퍼파라미터 그리드서치 적용)
- **Random Forest**: 비슷한 성능 수준

**향후 시도 예정:**
- XGBoost, LightGBM (부스팅 계열)
- 앙상블 방법론
- 간단한 DNN (과적합 주의)

## 📊 현재 성능
- **SVM 최고 성능**: 68-72%
- **Random Forest**: 비슷한 수준
- **목표 성능**: 90% 이상

## 📁 파일 구조
```
├── data/
│   ├── all_features_left.csv      # 좌수 전체 특성
│   ├── all_features_right.csv     # 우수 전체 특성
│   ├── labels_all_left.csv        # 좌수 라벨
│   ├── labels_all_right.csv       # 우수 라벨
│   ├── features_left.csv          # 좌수 선택 특성
│   └── features_right.csv         # 우수 선택 특성
├── eda_feature_selection.py       # EDA 및 특성 선택
├── svm_classification.py          # SVM 분류 및 시각화
├── randomforest_classification.py # Random Forest 분류
└── README.md
```

## 🛠️ 기술 스택
- **언어**: Python 3.11
- **주요 라이브러리**: 
  - scikit-learn (머신러닝)
  - pandas, numpy (데이터 처리)
  - matplotlib, seaborn (시각화)
- **모델**: SVM, Random Forest, (XGBoost 예정)

## 📈 진행 상황
- [x] 데이터 전처리 및 EDA 완료
- [x] 특성 선택 (198개 → 8-10개)
- [x] SVM 모델링 및 하이퍼파라미터 최적화
- [x] Random Forest 모델링
- [ ] 부스팅 계열 모델 (XGBoost, LightGBM) 실험
- [ ] 앙상블 방법론 적용
- [ ] 목표 성능 90%
- [ ] 허/실 체질 분류 확장

## 🔬 연구 배경
한의학에서 체질 분류는 개인 맞춤형 치료의 핵심입니다. 전통적인 맥진은 숙련된 한의사의 주관적 판단에 의존하는데, 본 연구는 객관적이고 정량적인 맥진 데이터를 통해 체질 분류의 자동화 가능성을 탐구합니다.

## ⚠️ 주의사항
- 작은 데이터셋 (90-95명)으로 인한 과적합 주의
- 의료 데이터 특성상 높은 정확도 요구
- 한의학 도메인 지식과의 연계 필요