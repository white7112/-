from modules.data_loader import get_prepared_data, get_class_names
from modules.models import evaluate_initial_models, tune_best_model
from modules.visualizer import plot_overfitting_analysis
from sklearn.metrics import f1_score, precision_score, recall_score
import os

def main():
    DATA_PATH = os.path.join('data', 'vanet_traffic_data.csv')
    TARGET = 'optimal_route_chosen'
    
    X_train, X_test, y_train, y_test = get_prepared_data(DATA_PATH, TARGET)
    class_names = get_class_names(DATA_PATH, TARGET)
    
    pipelines = evaluate_initial_models(X_train, X_test, y_train, y_test)
    
    model_names = list(pipelines.keys())
    train_scores = []
    test_scores = []
    
    print("\n" + "=" * 85)
    print("      [단계 1 & 2] 베이스라인 및 개선 모델 연산")
    print("=" * 85)
    
    best_test_f1 = -1
    best_model_name = ""
    
    for name in model_names:
        model = pipelines[name]
        
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        tr_f1 = f1_score(y_train, train_pred, average='macro', zero_division=0)
        te_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
        te_prec = precision_score(y_test, test_pred, average='macro', zero_division=0)
        te_rec = recall_score(y_test, test_pred, average='macro', zero_division=0)
        
        train_scores.append(tr_f1)
        test_scores.append(te_f1)
        
        if te_f1 > best_test_f1:
            best_test_f1 = te_f1
            best_model_name = name
            
        gap = tr_f1 - te_f1
        overfit_status = "과적합 경향성 확인" if gap > 0.02 else "안정적 일반화 상태"
        
        print(f"[{name} 모델 결과]")
        print(f" - 데이터셋 클래스    : {class_names}")
        print(f" - 훈련 세트 F1-Score : {tr_f1:.4f}")
        print(f" - 검증 세트 F1-Score : {te_f1:.4f} (정밀도: {te_prec:.4f} / 재현율: {te_rec:.4f})")
        print(f" - 데이터셋 성능 격차  : {gap:.4f} -> [{overfit_status}]")
        print("-" * 85)
        
    print("\n" + "=" * 85)
    print(f"      [단계 3 & 4] 하이퍼파라미터 최적화 (선정 모델: {best_model_name})")
    print("=" * 85)
    
    tuned_model, best_params = tune_best_model(X_train, y_train, best_model_name)
    
    tuned_train_pred = tuned_model.predict(X_train)
    tuned_test_pred = tuned_model.predict(X_test)
    
    tuned_tr_f1 = f1_score(y_train, tuned_train_pred, average='macro', zero_division=0)
    tuned_te_f1 = f1_score(y_test, tuned_test_pred, average='macro', zero_division=0)
    
    print(f"\n[최적화 파라미터 반환 조율 결과]")
    print(f" - 튜닝 후 훈련 F1  : {tuned_tr_f1:.4f}")
    print(f" - 튜닝 후 테스트 F1: {tuned_te_f1:.4f}")
    print(f" - 최종 파일 저장 완료")
    print("=" * 85)
    
    plot_overfitting_analysis(train_scores, test_scores, model_names)

if __name__ == "__main__":
    main()