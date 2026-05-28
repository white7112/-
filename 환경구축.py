"""
================================================================================
data_loader.py  -  환경 구축 담당 모듈
================================================================================
[이 파일이 하는 일을 한 줄로]
  CSV 파일에서 데이터를 읽어와서, 모델이 바로 학습할 수 있는 상태로 만들어준다.

[왜 따로 모듈로 만들었나]
  데이터 준비 과정은 한 번만 제대로 짜두면, 모든 팀원이 똑같이 사용할 수 있다.
  "각자 자기 방식으로 전처리"를 하면 모델 비교가 공정하지 않다.
  (예: A는 결측치 0으로 채우고, B는 평균으로 채우면 어느 모델이 좋은지 알 수 없다)
================================================================================
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# ============================================================
# [팀원과의 약속] 절대 바뀌지 않을 설정값
# ============================================================
# 이 값들을 함수 안이 아닌 파일 맨 위에 둔 이유:
#   - 누가 봐도 한눈에 보이게 하려고
#   - 나중에 데이터셋이 바뀌면 여기 한 줄만 고치면 되도록
RANDOM_SEED = 42         # 시드 고정 → 누가 돌려도 같은 결과
TEST_SIZE = 0.2          # 8:2 분할 (계획서 명시)
RESULT_DIR = "./result"  # 학습된 모델, 전처리 도구 저장 위치


def get_prepared_data(data_path, target_col="label"):
    """
    데이터를 준비해서 (X_train, X_test, y_train, y_test) 4개를 반환한다.

    [팀원 사용법]
        from data_loader import get_prepared_data
        X_train, X_test, y_train, y_test = get_prepared_data("data/vanet_traffic_data.csv")

    [내부에서 일어나는 일 - 순서대로]
        1) CSV 파일을 읽는다
        2) 학습에 쓸모없는 컬럼(timestamp, road_segment_id)을 버린다
        3) 결측치를 채운다 (수치형은 중앙값, 범주형은 최빈값)
        4) 범주형 피처를 숫자로 바꾼다 (Label Encoding)
        5) 타겟(정답 라벨)도 숫자로 바꾼다
        6) 8:2 비율로 train/test로 나눈다
        7) 수치 스케일을 맞춰준다 (StandardScaler)
        8) 나중에 다시 쓸 도구들(scaler, encoder)을 파일로 저장한다

    Args:
        data_path (str): CSV 파일 경로
        target_col (str): 정답 라벨 컬럼명 (기본값: "label")

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # -------- [1] CSV 읽기 --------
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()  # 컬럼명 앞뒤 공백 제거 (CSV 만든 사람이 실수했을 때 대비)

    # 타겟 컬럼이 진짜 있는지 먼저 확인 → 없으면 친절한 에러
    if target_col not in df.columns:
        raise KeyError(
            f"타겟 컬럼 '{target_col}'이 CSV에 없습니다. "
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    # -------- [2] 학습에 쓸모없는 컬럼 제거 --------
    # timestamp(시간), road_segment_id(도로 ID)는 학습에 도움 안 됨
    # 시간 자체로는 패턴을 못 잡고, 도로 ID는 그냥 고유번호일 뿐
    cols_to_drop = ["timestamp", "road_segment_id"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # -------- [3] 피처(X)와 정답(y) 분리 --------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # -------- [4] 결측치(빈 값) 처리 --------
    # 왜 dropna()로 행을 지우지 않나?
    #   결측치 행을 통째로 지우면 데이터가 줄어들어 학습이 약해진다.
    #   계획서에도 "수치형=중앙값, 범주형=최빈값"으로 명시되어 있다.
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns

    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())      # 중앙값으로 채움
    for col in categorical_cols:
        X[col] = X[col].fillna(X[col].mode()[0])     # 가장 많이 나온 값으로 채움

    # -------- [5] 범주형 피처를 숫자로 변환 --------
    # 모델은 "Sunny", "Rainy" 같은 문자를 못 알아듣는다 → 0, 1로 바꿔야 함
    # 인코더를 dict에 저장해두는 이유:
    #   나중에 새 데이터("Sunny")가 들어왔을 때 똑같이 0으로 바꿔야 일관성 유지
    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le

    # -------- [6] 타겟(정답)도 숫자로 변환 --------
    # 예: "Free-flow" → 0, "Gridlock" → 1, "Heavy" → 2, "Moderate" → 3
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y.astype(str))

    # -------- [7] 8:2 분할 (stratify로 클래스 비율 유지) --------
    # stratify=y_encoded의 의미:
    #   원본에서 Free-flow가 30%였다면 train/test에도 30%씩 들어가게 한다
    #   안 하면 한쪽에 특정 클래스가 너무 몰릴 수 있음
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    # -------- [8] 스케일링 (수치 범위 맞춰주기) --------
    # 예: 속도는 0~120, 거리는 0~10000 → 그대로 쓰면 거리에 모델이 휘둘림
    # StandardScaler는 평균 0, 표준편차 1로 맞춰준다
    # 중요: scaler는 train에만 fit, test에는 transform만 → "데이터 누수" 방지
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------- [9] 도구 저장 (가장 중요한 부분) --------
    # 왜 저장해야 하나?
    #   나중에 학습된 모델로 새 데이터를 예측할 때, "학습할 때 썼던 scaler"로
    #   똑같이 변환해야 결과가 맞다. 매번 새로 만들면 결과가 다르게 나온다.
    os.makedirs(RESULT_DIR, exist_ok=True)
    joblib.dump(scaler, f"{RESULT_DIR}/scaler.pkl")
    joblib.dump(le_y, f"{RESULT_DIR}/label_encoder_y.pkl")
    joblib.dump(feature_encoders, f"{RESULT_DIR}/feature_encoders.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test


def get_class_names(data_path="./data/vanet_traffic_data.csv", target_col="label"):
    """
    저장된 label encoder를 불러와서 클래스 이름 리스트를 돌려준다.

    [사용 예]
        names = get_class_names()
        # → ['Free-flow', 'Gridlock', 'Heavy', 'Moderate']
        # 모델이 예측값 2를 내놓으면 → names[2] = 'Heavy'

    [왜 필요한가]
        모델은 결과를 숫자(0,1,2,3)로만 내놓는다. 사람이 보기엔 "Heavy"가 직관적.
        그래프 라벨, 혼동행렬, 발표 자료 만들 때 거의 무조건 필요하다.
    """
    encoder_path = f"{RESULT_DIR}/label_encoder_y.pkl"
    if not os.path.exists(encoder_path):
        # 아직 학습 안 했으면 데이터 한 번 불러서 저장부터
        get_prepared_data(data_path, target_col)
    le_y = joblib.load(encoder_path)
    return list(le_y.classes_)


# ============================================================
# 단독 실행: 환경 구축이 잘 됐는지 확인
# ============================================================
# 이 파일을 직접 실행하면 (python data_loader.py) 아래 코드가 돈다.
# 다른 곳에서 import만 할 때는 안 돈다.
if __name__ == "__main__":
    print("=" * 60)
    print("환경 구축 점검")
    print("=" * 60)

    DATA_PATH = "./data/vanet_traffic_data.csv"
    X_train, X_test, y_train, y_test = get_prepared_data(DATA_PATH)

    print(f"X_train 형태 : {X_train.shape}  (행, 열)")
    print(f"X_test  형태 : {X_test.shape}")
    print(f"y_train 형태 : {y_train.shape}")
    print(f"y_test  형태 : {y_test.shape}")
    print(f"클래스 목록  : {get_class_names(DATA_PATH)}")
    print()

    print("저장된 도구 확인:")
    for f in ["scaler.pkl", "label_encoder_y.pkl", "feature_encoders.pkl"]:
        path = f"{RESULT_DIR}/{f}"
        ok = "OK" if os.path.exists(path) else "MISSING"
        print(f"  [{ok}] {path}")
    print()
    print("환경 구축 완료. 팀원들은 아래처럼 쓰면 됩니다:")
    print("  from data_loader import get_prepared_data")
    print("  X_train, X_test, y_train, y_test = get_prepared_data('data/vanet_traffic_data.csv')")