import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

RANDOM_SEED = 42
TEST_SIZE = 0.2
RESULT_DIR = "./result"

def get_prepared_data(data_path, target_col="label"):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    if target_col not in df.columns:
        raise KeyError(
            f"타겟 컬럼 '{target_col}'이 CSV에 없습니다. "
            f"사용 가능한 컬럼: {list(df.columns)}"
        )

    df = df.dropna(subset=[target_col])

    cols_to_drop = ["timestamp", "road_segment_id"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 숫자형 데이터와 문자형(범주형) 데이터는 빈칸을 채우는 방법이 달라서 오류가 나기 때문에 2개의 타입을 분리
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns

    # 숫자 데이터의 빈칸은 '중앙값'으로 대체한다.
    # 평균값은 말도 안 되게 크거나 작은 이상치(예: 폭주하는 차량 속도)에 쉽게 왜곡되지만,
    # 크기순으로 나열했을 때 딱 중간에 위치한 '중앙값'은 가장 평범하고 안전하여 데이터의 왜곡을 막음
    for col in numeric_cols:
        median_val = X[col].median()
        if pd.isna(median_val):
            median_val = 0
        X[col] = X[col].fillna(median_val)

    # 글자(범주형) 데이터는 숫자가 아니라서 더하거나 나눌 수 없어 평균을 낼 수 없음
    # 그래서 데이터 중 가장 흔하게 등장한 '최빈값'(가장 빈도가 높은 글자)을 채워 넣는 것이 
    # 통계학적으로 데이터의 본질을 헤치지 않을 것으로 예측.
    for col in categorical_cols:
        mode_series = X[col].mode()
        if not mode_series.empty:
            X[col] = X[col].fillna(mode_series[0])
        else:
            X[col] = X[col].fillna("Unknown")

    # 글자를 읽지 못하는 AI를 위해 문자형 데이터를 정수(0, 1, 2...)로 1:1 매핑(번역)하여 수학적 연산이 가능하도록 변환
    # 이때 컬럼별로 생성된 독립적인 번역 규칙(LabelEncoder)들을 'feature_encoders' 박스에 담아두고
    # 배포 환경이나 향후 새로운 데이터가 들어왔을 때도 완전히 동일한 기준으로 글자를 숫자로 바꾸기 위한 장치임
    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le

    
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y.astype(str))

    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(RESULT_DIR, exist_ok=True)
    joblib.dump(scaler, f"{RESULT_DIR}/scaler.pkl")
    joblib.dump(le_y, f"{RESULT_DIR}/label_encoder_y.pkl")
    joblib.dump(feature_encoders, f"{RESULT_DIR}/feature_encoders.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test

def get_class_names(data_path="./data/vanet_traffic_data.csv", target_col="label"):
    encoder_path = f"{RESULT_DIR}/label_encoder_y.pkl"
    if not os.path.exists(encoder_path):
        get_prepared_data(data_path, target_col)
    le_y = joblib.load(encoder_path)
    return list(le_y.classes_)