import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from sklearn.metrics import confusion_matrix

# 불필요한 경고 메시지 출력 방지
warnings.filterwarnings("ignore", category=FutureWarning)

# 한글 폰트 설정 및 마이너스 기호 처리 (그래프 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# [유틸리티] 시각적 출력 폭을 계산하는 함수 (한글 2칸, 영문/숫자 1칸 계산)
# 콘솔 출력 시 정렬이 틀어지는 근본 원인(한글 폭 불일치)을 해결하기 위함
def get_display_width(text):
    width = 0
    # 문자열 내 각 문자의 유니코드 속성을 확인하여 폭을 합산
    for char in str(text):
        # East Asian Width 속성이 넓은 문자인지 확인
        if unicodedata.east_asian_width(char) in ('W', 'F'):
            width += 2
        else:
            width += 1
    return width

# [유틸리티] 계산된 폭을 바탕으로 공백을 넣어 정렬하는 함수
def format_cell(text, width, align='left'):
    text = str(text)
    d_width = get_display_width(text)
    # 텍스트의 실제 출력 폭을 계산하여 부족한 만큼 공백 추가
    padding = max(0, width - d_width)
    if align == 'left':
        return text + (' ' * padding)
    else:
        return (' ' * padding) + text

# [1단계] 모델별 성능 비교표 출력 함수
def print_initial_comparison(results):
    print("\n" + "="*95)
    print(" [1단계] 기본 모델별 초기 성능 다각도 비교")
    print("="*95)
    
    # 표 헤더 출력 (format_cell로 고정 폭 적용)
    h1 = format_cell("모델명", 20)
    h2 = format_cell("데이터", 14)
    h3 = format_cell("정확도", 12, 'right')
    h4 = format_cell("정밀도", 12, 'right')
    h5 = format_cell("재현율", 12, 'right')
    h6 = format_cell("F1-스코어", 12, 'right')
    print(f"{h1}{h2}{h3}{h4}{h5}{h6}")
    print("-" * 95)
    
    # 출력용 모델명 매핑
    name_map = {"Logistic": "로지스틱 회귀", "DecisionTree": "결정 트리", "RandomForest": "랜덤 포레스트", "SVC": "SVC"}
    
    for res in results:
        m_name = name_map.get(res['name'], res['name'])
        # 훈련 데이터 행 (format_cell을 사용하여 완벽 수직 정렬)
        row1 = f"{format_cell(m_name, 20)}{format_cell('훈련', 14)}{format_cell(f'{res['tr_acc']:.4f}', 12, 'right')}{format_cell(f'{res['tr_prec']:.4f}', 12, 'right')}{format_cell(f'{res['tr_rec']:.4f}', 12, 'right')}{format_cell(f'{res['tr_f1']:.4f}', 12, 'right')}"
        # 테스트 데이터 행
        row2 = f"{format_cell('', 20)}{format_cell('테스트', 14)}{format_cell(f'{res['te_acc']:.4f}', 12, 'right')}{format_cell(f'{res['te_prec']:.4f}', 12, 'right')}{format_cell(f'{res['te_rec']:.4f}', 12, 'right')}{format_cell(f'{res['te_f1']:.4f}', 12, 'right')}"
        print(row1)
        print(row2)
        print("-" * 95)
        
    print("\n[💡 모델 선정 통계적 근거 분석]")
    # 과적합 및 앙상블 분석 내용 복구
    print(" - 결정 트리는 단일 구조상 데이터 노이즈에 과적합(Overfitting)될 위험이 상대적으로 높습니다.")
    print(" - 랜덤 포레스트는 앙상블 기법으로 모델의 분산(Variance)을 줄여 일반화 성능을 높입니다.")
    print(" - 따라서 혼잡한 VANET 교통 환경에서 안정적인 '랜덤 포레스트'를 최종 모델로 선정합니다.")
    print("="*95)

# [2단계] 최적 모델 상세 결과 출력 함수
def print_detailed_report(report_dict):
    print("\n[상세 분류 보고서]")
    h1 = format_cell("클래스", 24)
    h2 = format_cell("정밀도", 16, 'right')
    h3 = format_cell("재현율", 16, 'right')
    h4 = format_cell("F1-스코어", 16, 'right')
    h5 = format_cell("개수", 12, 'right')
    print(f"{h1}{h2}{h3}{h4}{h5}")
    print("-" * 84)
    
    # 각 분류 지표 반복 출력
    for k, v in report_dict.items():
        if k.isdigit():
            row = f"{format_cell(k, 24)}{format_cell(f'{v['precision']:.4f}', 16, 'right')}{format_cell(f'{v['recall']:.4f}', 16, 'right')}{format_cell(f'{v['f1-score']:.4f}', 16, 'right')}{format_cell(int(v['support']), 12, 'right')}"
            print(row)
            
    print("-" * 84)
    # 정확도 및 평균 지표 출력
    print(f"{format_cell('정확도(Accuracy)', 56)}{format_cell(f'{report_dict['accuracy']:.4f}', 16, 'right')}")
    print(f"{format_cell('매크로 평균(Macro Avg)', 24)}{format_cell(f'{report_dict['macro avg']['precision']:.4f}', 16, 'right')}{format_cell(f'{report_dict['macro avg']['recall']:.4f}', 16, 'right')}{format_cell(f'{report_dict['macro avg']['f1-score']:.4f}', 16, 'right')}")
    print(f"{format_cell('가중 평균(Weighted Avg)', 24)}{format_cell(f'{report_dict['weighted avg']['precision']:.4f}', 16, 'right')}{format_cell(f'{report_dict['weighted avg']['recall']:.4f}', 16, 'right')}{format_cell(f'{report_dict['weighted avg']['f1-score']:.4f}', 16, 'right')}")
    print("=" * 84)

# 결과 시각화 (혼동 행렬 및 막대 그래프)
def plot_final_results(y_test, y_pred, metrics, model_name):
    plt.figure(figsize=(14, 6))
    
    # 그래프 1: 혼동 행렬 시각화
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'혼동 행렬 ({model_name})')
    plt.xlabel('예측값'); plt.ylabel('실제값')
    
    # 그래프 2: 성능 지표 막대 차트
    plt.subplot(1, 2, 2)
    labels = ['정확도', '정밀도', '재현율', 'F1-스코어']
    sns.barplot(x=labels, y=metrics, hue=labels, palette='viridis', legend=False)
    plt.title(f'최종 성능 지표 ({model_name})')
    plt.ylim(0.0, 1.05)
    
    # 막대 상단 수치 표기
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f"{v:.5f}", ha='center', fontsize=10, weight='bold')
        
    plt.tight_layout()
    plt.savefig("./result/performance_graph.png")
    plt.show()

# 과적합 분석 그래프 (학습 vs 테스트)
def plot_overfitting_analysis(train_scores, test_scores, model_names):
    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 학습 점수 막대
    rects1 = ax.bar(x - width/2, train_scores, width, label='Train Score', color='#4c72b0')
    # 테스트 점수 막대
    rects2 = ax.bar(x + width/2, test_scores, width, label='Test Score', color='#dd8452')
    
    ax.set_ylabel('F1-Score'); ax.set_title('Model Overfitting Analysis')
    ax.set_xticks(x); ax.set_xticklabels(model_names); ax.legend(); ax.set_ylim(0.0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 막대 상단 수치 표기
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.5f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=9)
    plt.tight_layout()
    plt.show()