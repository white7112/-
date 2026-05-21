import numpy as np
import pandas as pd
import joblib 
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC 

# 1. 환경 설정 및 폴더 생성
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
os.makedirs("./result", exist_ok=True) # result 폴더가 없으면 생성

# 2. 데이터 로드
data_path = "./data/vanet_traffic_data.csv"
target_column = "label"

df = pd.read_csv(data_path)
X = df.drop(columns=[target_column])
y = df[target_column]

# 3. 데이터 전처리
numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(include=['object', 'string']).columns

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

if y.dtype == "object":
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)

# 4. 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# [공통 최적화] 학습 속도를 위해 20% 샘플링
X_train_small, _, y_train_small, _ = train_test_split(
    X_train_scaled, y_train, train_size=0.2, random_state=RANDOM_SEED
)

# 5. 모델 평가 함수
def evaluate_model(model, X_tr, X_te, y_tr, y_te, name):
    print(f"[{name}] 학습 중...")
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="weighted")
    print(f"[{name}] 결과 -> 정확도: {acc:.4f}, F1-Score: {f1:.4f}")
    return model

# 6. 모델 실험
print("\n--- [1] 베이스라인 실험 ---")
lr_model = evaluate_model(LogisticRegression(max_iter=1000), X_train_scaled, X_test_scaled, y_train, y_test, "Logistic Regression")
dt_model = evaluate_model(DecisionTreeClassifier(), X_train_scaled, X_test_scaled, y_train, y_test, "Decision Tree")

print("\n--- [2] 개선 모델 실험 (최적화 적용) ---")
rf_model = evaluate_model(
    RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=RANDOM_SEED), 
    X_train_small, X_test_scaled, y_train_small, y_test, "Random Forest"
)
svc_model = evaluate_model(LinearSVC(dual="auto", random_state=RANDOM_SEED), X_train_small, X_test_scaled, y_train_small, y_test, "Linear SVC")

# 7. result 폴더에 모델 저장
print("\n[완료] 모든 모델을 'result/' 폴더에 저장합니다.")
joblib.dump(lr_model, "./result/lr_model.pkl")
joblib.dump(rf_model, "./result/rf_model.pkl")
joblib.dump(svc_model, "./result/svc_model.pkl")