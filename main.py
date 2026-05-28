import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from modules.data_loader import get_prepared_data
from modules.models import run_hyperparameter_tuning, evaluate_initial_models
from modules.visualizer import print_initial_comparison, print_detailed_report, plot_final_results

def main():
    # 1. 데이터 로드 및 전처리
    data_path = "./data/vanet_traffic_data.csv"
    X_train, X_test, y_train, y_test = get_prepared_data(data_path, target_col="optimal_route_chosen")
    
    # 2. [1단계] 후보 모델 성능 평가
    # 로지스틱 회귀, 결정 트리, 랜덤 포레스트, SVC를 포함한 초기 성능 비교 수행
    initial_models = evaluate_initial_models(X_train, X_test, y_train, y_test)
    comparison_results = []
    
    for name, model in initial_models.items():
        # 훈련 및 테스트 데이터에 대한 예측 수행
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # 각 모델의 성능 지표 계산 및 리스트 추가 (SVC 자동 포함)
        comparison_results.append({
            "name": name,  # models.py에서 정의한 키 이름 그대로 전달 (visualizer에서 매핑)
            "tr_acc": accuracy_score(y_train, train_pred),
            "tr_prec": precision_score(y_train, train_pred, average='macro'),
            "tr_rec": recall_score(y_train, train_pred, average='macro'),
            "tr_f1": f1_score(y_train, train_pred, average='macro'),
            "te_acc": accuracy_score(y_test, test_pred),
            "te_prec": precision_score(y_test, test_pred, average='macro'),
            "te_rec": recall_score(y_test, test_pred, average='macro'),
            "te_f1": f1_score(y_test, test_pred, average='macro')
        })
        
    # 비교 결과를 터미널에 가독성 높게 출력
    print_initial_comparison(comparison_results)
    
    # 3. [2단계] 최적 모델 선정 및 하이퍼파라미터 튜닝
    print("\n" + "="*85)
    print(" [2단계] 최종 선정 모델 하이퍼파라미터 튜닝")
    print("="*85)
    best_model, best_params, name = run_hyperparameter_tuning(X_train, y_train)
    
    # 튜닝된 최적 모델 예측
    y_pred = best_model.predict(X_test)
    metrics = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='macro'),
        recall_score(y_test, y_pred, average='macro'),
        f1_score(y_test, y_pred, average='macro')
    ]
    
    # 결과 출력
    print(f"\n[최종 선택된 최적 모델: {name}]")
    print(best_model)
    print("\n[최적 하이퍼파라미터]")
    print(best_params)
    
    # 4. 결과 분석 및 모델 저장
    # 상세 분류 보고서 출력
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    print_detailed_report(report_dict)
    
    # 최적 모델 파일 저장
    os.makedirs("./result", exist_ok=True)
    joblib.dump(best_model, "./result/best_model.pkl")
    print("\n최적 모델이 ./result/best_model.pkl 에 저장되었습니다.")
    
    # 성능 시각화
    plot_final_results(y_test, y_pred, metrics, name)

if __name__ == "__main__":
    main()

# Macro Average - 모든 클래스를 동일한 중요도로 평가하는 지표입니다. 데이터 개수와 상관없이 개별 클래스의 성능을 산술 평균하므로,
# 클래스 간 데이터 불균형이 심할 때 모델이 특정 다수 클래스에만 치우치지 않았는지 확인하는 균형점 역할을 합니다. 
# 이를 통해 데이터가 적은(Rare) 클래스에서도 모델이 제대로 학습하고 있는지 진단하여, 모델의 편향 없는 일반화 성능을 검증합니다.

# 가중 평균 (Weighted Average) - 각 클래스의 데이터 개수(Support)를 가중치로 반영하여 산출한 평균 성능 지표입니다. 
# 실제 데이터가 많은 클래스의 성능 비중을 높여서 계산하므로, 우리 프로젝트의 실제 데이터 분포를 반영한 '실체감 성능'을 보여줍니다. 
# 따라서 전체적인 모델의 예측 신뢰도를 실무 환경에 맞춰 가장 현실적으로 판단하고 평가할 때 핵심 근거로 사용합니다.

# 혼동 행렬 - 모델의 예측값과 실제값을 교차 표기하여, 모델이 어떤 클래스를 다른 클래스로 오인하고 있는지 오류 패턴을 시각화합니다. 
# 단순한 정확도 수치로는 알 수 없는 '구체적으로 어디서 틀리는가'에 대한 정보를 제공하므로, 특정 클래스에 대한 혼동(Confusion)을 정밀하게 진단할 수 있습니다. 
# 이 행렬을 분석함으로써 데이터 자체의 문제인지 모델 구조의 한계인지 파악하여, 후속 튜닝의 명확한 개선 방향을 수립하는 진단 근거가 됩니다.