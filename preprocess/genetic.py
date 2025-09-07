import pandas as pd
import numpy as np
import random
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=== 유전 알고리즘 기반 특성 선택 ===")
print("목표: 80% 성능 달성을 위한 최적 특성 조합 찾기")

class GeneticFeatureSelector:
    def __init__(self, X, y, model_type='svm', population_size=50, 
                 generations=100, mutation_rate=0.1, crossover_rate=0.8,
                 min_features=3, max_features=15, cv_folds=5):
        """
        유전 알고리즘 기반 특성 선택기
        
        Parameters:
        - X, y: 데이터와 라벨
        - model_type: 'svm', 'rf', 'lr' 중 선택
        - population_size: 개체군 크기
        - generations: 세대 수
        - mutation_rate: 돌연변이 확률
        - crossover_rate: 교배 확률
        - min_features, max_features: 특성 개수 범위
        - cv_folds: 교차검증 폴드 수
        """
        self.X = X
        self.y = y
        self.feature_names = X.columns.tolist()
        self.n_features = len(self.feature_names)
        
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.min_features = min_features
        self.max_features = max_features
        self.cv_folds = cv_folds
        
        # 모델 설정
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', random_state=42)
        elif model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'lr':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        self.scaler = StandardScaler()
        
        # 결과 저장
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_individual = None
        self.best_fitness = 0
        
    def create_individual(self):
        """개체 생성 (특성 선택 마스크)"""
        n_selected = random.randint(self.min_features, self.max_features)
        individual = [0] * self.n_features
        selected_indices = random.sample(range(self.n_features), n_selected)
        for idx in selected_indices:
            individual[idx] = 1
        return individual
    
    def create_population(self):
        """초기 개체군 생성"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def evaluate_fitness(self, individual):
        """적합도 평가 (교차검증 성능)"""
        selected_features = [i for i, gene in enumerate(individual) if gene == 1]
        
        if len(selected_features) == 0:
            return 0
        
        X_selected = self.X.iloc[:, selected_features]
        
        try:
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # 교차검증 수행
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(self.model, X_scaled, self.y, cv=cv, scoring='accuracy')
            
            fitness = scores.mean()
            
            # 특성 개수에 따른 패널티 (너무 많은 특성 사용 방지)
            if len(selected_features) > self.max_features * 0.8:
                fitness *= 0.95
                
            return fitness
            
        except Exception as e:
            return 0
    
    def tournament_selection(self, population, fitnesses, tournament_size=3):
        """토너먼트 선택"""
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
            selected.append(population[winner_idx].copy())
        return selected
    
    def crossover(self, parent1, parent2):
        """교배 (단일점 교차)"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # 특성 개수 제한 확인
        self._adjust_feature_count(child1)
        self._adjust_feature_count(child2)
        
        return child1, child2
    
    def _adjust_feature_count(self, individual):
        """특성 개수를 범위 내로 조정"""
        n_selected = sum(individual)
        
        if n_selected < self.min_features:
            # 특성 추가
            available_indices = [i for i, gene in enumerate(individual) if gene == 0]
            to_add = min(self.min_features - n_selected, len(available_indices))
            add_indices = random.sample(available_indices, to_add)
            for idx in add_indices:
                individual[idx] = 1
                
        elif n_selected > self.max_features:
            # 특성 제거
            selected_indices = [i for i, gene in enumerate(individual) if gene == 1]
            to_remove = n_selected - self.max_features
            remove_indices = random.sample(selected_indices, to_remove)
            for idx in remove_indices:
                individual[idx] = 0
    
    def mutate(self, individual):
        """돌연변이"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # 비트 플립
        
        # 특성 개수 제한 확인
        self._adjust_feature_count(mutated)
        
        return mutated
    
    def evolve(self):
        """진화 과정 실행"""
        print(f"유전 알고리즘 시작:")
        print(f"- 개체군 크기: {self.population_size}")
        print(f"- 세대 수: {self.generations}")
        print(f"- 특성 범위: {self.min_features}-{self.max_features}개")
        print(f"- 교배율: {self.crossover_rate}, 돌연변이율: {self.mutation_rate}")
        
        # 초기 개체군 생성
        population = self.create_population()
        
        for generation in range(self.generations):
            # 적합도 평가
            fitnesses = [self.evaluate_fitness(individual) for individual in population]
            
            # 최고 개체 기록
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitnesses[max_fitness_idx]
                self.best_individual = population[max_fitness_idx].copy()
            
            # 통계 기록
            self.best_fitness_history.append(max(fitnesses))
            self.avg_fitness_history.append(np.mean(fitnesses))
            
            # 진행상황 출력
            if generation % 10 == 0 or generation == self.generations - 1:
                selected_count = sum(self.best_individual) if self.best_individual else 0
                print(f"세대 {generation:3d}: 최고 적합도 {self.best_fitness:.4f} "
                      f"(특성 {selected_count}개), 평균 {np.mean(fitnesses):.4f}")
            
            if generation == self.generations - 1:
                break
                
            # 선택
            selected_population = self.tournament_selection(population, fitnesses)
            
            # 교배 및 돌연변이
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[min(i+1, len(selected_population)-1)]
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return self.get_best_features()
    
    def get_best_features(self):
        """최적 특성 반환"""
        if self.best_individual is None:
            return []
        
        selected_indices = [i for i, gene in enumerate(self.best_individual) if gene == 1]
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        return {
            'features': selected_features,
            'feature_indices': selected_indices,
            'fitness': self.best_fitness,
            'n_features': len(selected_features)
        }
    
    def plot_evolution(self):
        """진화 과정 시각화"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.best_fitness_history, label='최고 적합도', color='red', linewidth=2)
        plt.plot(self.avg_fitness_history, label='평균 적합도', color='blue', alpha=0.7)
        plt.xlabel('세대')
        plt.ylabel('적합도 (정확도)')
        plt.title('유전 알고리즘 진화 과정')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        generation_improvements = []
        best_so_far = 0
        for fitness in self.best_fitness_history:
            if fitness > best_so_far:
                best_so_far = fitness
                generation_improvements.append(len(generation_improvements))
        
        if generation_improvements:
            improvement_values = [self.best_fitness_history[gen] for gen in generation_improvements]
            plt.scatter(generation_improvements, improvement_values, color='red', s=50, alpha=0.7)
            plt.plot(generation_improvements, improvement_values, color='red', alpha=0.5)
            plt.xlabel('세대')
            plt.ylabel('개선된 적합도')
            plt.title('성능 개선 지점')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# 데이터 로드
X = pd.read_csv('data/all_features_left.csv')  # 또는 features_right.csv로 변경
y = pd.read_csv('data/labels_all_left.csv')
y = y.astype(int)

# 라벨 처리
y_flat = np.ravel(y)
if set(np.unique(y_flat)) == {1, 2}:
    y_flat = y_flat - 1
    print("라벨 변환: {1,2} → {0,1}")

print(f"데이터 크기: {X.shape}")
print(f"클래스 분포: {np.bincount(y_flat)}")

# ===== CPU 사용량 제어 설정 =====
CPU_USAGE = 'medium'  # 'low', 'medium', 'high' 중 선택
print(f"CPU 사용 모드: {CPU_USAGE}")

import os
if CPU_USAGE == 'low':
    n_jobs_setting = 1
    population_size = 20
    generations = 30
    print("🐌 저사용 모드: 1코어, 작은 탐색 공간")
elif CPU_USAGE == 'medium':
    cpu_count = os.cpu_count() or 4
    n_jobs_setting = max(1, cpu_count // 2)
    population_size = 30
    generations = 50
    print(f"⚖️ 중간 사용 모드: {n_jobs_setting}코어")
else:
    n_jobs_setting = -1
    population_size = 50
    generations = 100
    print("🚀 고사용 모드: 모든 코어 사용")

# CPU 사용량에 따른 파라미터 설정
if CPU_USAGE == 'low':
    population_size = 20
    generations = 30
elif CPU_USAGE == 'medium':
    population_size = 30
    generations = 50
else:  # high
    population_size = 50
    generations = 100

# 유전 알고리즘 실행 (여러 모델로 시도)
models_to_try = ['svm', 'rf', 'lr']
results = {}

for model_type in models_to_try:
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} 모델로 유전 알고리즘 실행")
    print(f"{'='*60}")
    
    # 유전 알고리즘 설정 (CPU 사용량에 맞게 동적 조정)
    # 데이터 특성 수에 맞게 최대 특성 수 조정
    max_possible_features = min(50, X.shape[1] - 1)  # 전체 특성 수보다 작게 설정
    min_possible_features = min(3, max_possible_features)  # 최소값도 조정
    
    ga = GeneticFeatureSelector(
        X=X, 
        y=y_flat,
        model_type=model_type,
        population_size=population_size,  # CPU 모드에 따라 조정됨
        generations=generations,          # CPU 모드에 따라 조정됨
        mutation_rate=0.15,      # 높은 돌연변이율 (다양성 확보)
        crossover_rate=0.8,
        min_features=min_possible_features,  # 동적으로 조정
        max_features=max_possible_features,  # 동적으로 조정
        cv_folds=3               # CPU 절약을 위해 3-fold로 고정
    )
    
    start_time = time.time()
    best_result = ga.evolve()
    end_time = time.time()
    
    results[model_type] = best_result
    
    print(f"\n{model_type.upper()} 결과:")
    print(f"- 최고 성능: {best_result['fitness']:.4f} ({best_result['fitness']*100:.1f}%)")
    print(f"- 선택된 특성 수: {best_result['n_features']}개")
    print(f"- 소요 시간: {end_time - start_time:.1f}초")
    print(f"- 선택된 특성:")
    for i, feature in enumerate(best_result['features'], 1):
        print(f"  {i:2d}. {feature}")
    
    # 진화 과정 시각화
    ga.plot_evolution()

# 최종 결과 비교
print(f"\n{'='*60}")
print("최종 결과 비교")
print(f"{'='*60}")

# 결과 정렬 (성능순)
sorted_results = sorted(results.items(), key=lambda x: x[1]['fitness'], reverse=True)

print(f"\n성능 순위:")
for rank, (model, result) in enumerate(sorted_results, 1):
    print(f"{rank}. {model.upper()}: {result['fitness']:.4f} ({result['fitness']*100:.1f}%) "
          f"- {result['n_features']}개 특성")

# 최고 성능 모델 상세 정보
best_model, best_result = sorted_results[0]
print(f"\n최고 성능 모델: {best_model.upper()}")
print(f"성능: {best_result['fitness']:.4f} ({best_result['fitness']*100:.1f}%)")
print(f"선택된 특성 ({best_result['n_features']}개):")
for i, feature in enumerate(best_result['features'], 1):
    print(f"  {i:2d}. {feature}")

# 목표 달성 여부
target_accuracy = 0.80
if best_result['fitness'] >= target_accuracy:
    print(f"\n🎉 목표 달성! {best_result['fitness']*100:.1f}% >= 80%")
else:
    gap = target_accuracy - best_result['fitness']
    print(f"\n📊 목표까지 {gap*100:.1f}%p 부족 (현재: {best_result['fitness']*100:.1f}%)")

# 실제 테스트를 위한 코드 출력
print(f"\n💾 최적 설정 (복사용):")
print(f"best_features = {best_result['features']}")
print(f"best_model_type = '{best_model}'")
print(f"expected_performance = {best_result['fitness']:.4f}")

print(f"\n✅ 유전 알고리즘 기반 특성 선택 완료!")