import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def evaluate_initial_models(X_train, X_test, y_train, y_test):
    
    # 초기 후보 모델들을 학습시켜 성능을 비교하기 위한 기본 설정 함수입니다.
    # 로지스틱 회귀, 결정 트리, 랜덤 포레스트, SVC 4종을 학습합니다.
    
    models = {
        "Logistic": LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=None, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "SVC": SVC(probability=True, random_state=42) # 확률 예측 기능 활성화 (추후 분석 대비)
    }
    
    pipelines = {}
    for name, model in models.items():
        # 각 모델별로 학습(fit)을 수행하여 파이프라인 딕셔너리에 저장
        model.fit(X_train, y_train)
        pipelines[name] = model
        
    return pipelines

def run_hyperparameter_tuning(X_train, y_train):
    
    # 랜덤 포레스트와 SVC 모델을 대상으로 하이퍼파라미터 튜닝을 수행합니다.
    # RandomizedSearchCV를 사용하여 최적의 조합을 찾고, 성능이 더 좋은 모델을 최종 선정합니다.
    
    # 1. 랜덤 포레스트 하이퍼파라미터 탐색 범위 설정
    rf_param = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"]
    }
    # 3-Fold 교차 검증을 통해 최적 조합 탐색
    rf_search = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_param, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    # 2. SVC 하이퍼파라미터 탐색 범위 설정
    svc_param = {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1], "kernel": ["rbf", "linear"]}
    svc_search = RandomizedSearchCV(SVC(random_state=42), svc_param, n_iter=5, cv=3, random_state=42, n_jobs=-1)
    svc_search.fit(X_train, y_train)

    # 3. 모델 비교 및 최종 선정
    # 두 모델의 최고 평균 점수(best_score_)를 비교하여 더 우수한 모델과 파라미터를 반환합니다.
    if rf_search.best_score_ >= svc_search.best_score_:
        return rf_search.best_estimator_, rf_search.best_params_, "Random Forest (랜덤 포레스트)"
    return svc_search.best_estimator_, svc_search.best_params_, "SVC (서포트 벡터 분류기)"