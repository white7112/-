import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def evaluate_initial_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic": LogisticRegression(C=0.1, solver='lbfgs', max_iter=2000, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=None, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    }
    
    pipelines = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pipelines[name] = model
        
    return pipelines

def tune_best_model(X_train, y_train, best_model_name):
    best_params = {
        'n_estimators': 150,
        'max_depth': 15,
        'min_samples_split': 2
    }
    
    tuned_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        random_state=42,
        n_jobs=-1
    )
    tuned_model.fit(X_train, y_train)
    
    os.makedirs('./result', exist_ok=True)
    joblib.dump(tuned_model, './result/final_model.pkl')
    
    return tuned_model, best_params