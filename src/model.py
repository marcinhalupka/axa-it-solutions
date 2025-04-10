from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import optuna

def objective(trial, X_train, y_train):
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    param = {
        'objective': 'binary',
        'n_jobs': -1,
        'n_estimators': trial.suggest_int('n_estimators', 10, 200),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = LGBMClassifier(**param, verbosity=-1)
    model.fit(X_train_sub, y_train_sub)
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

def tune_hyperparameters(X_train, y_train, n_trials=20):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    return study.best_params

def train_final_model(X_train, y_train, best_params):
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    # Optionally display the confusion matrix:
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    return metrics
