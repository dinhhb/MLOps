import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import mlflow
import mlflow.sklearn
import optuna

mlflow.set_experiment("my-project")

DATA_PATH = "tcga_gdc.csv"

CLINICAL_CATEGORICAL_COLS = [
    'ETHNICITY', 'ICD_10', 'MORPHOLOGY', 'SAMPLE_TYPE',
    'PATH_M_STAGE', 'PATH_N_STAGE', 'PATH_STAGE', 'PATH_T_STAGE',
    'PRIMARY_SITE_PATIENT', 'PRIOR_MALIGNANCY', 'PRIOR_TREATMENT', 'RACE',
    'SEX',
]
CLINICAL_NUMERIC_COLS = ['AGE', 'TMB_NONSYNONYMOUS']
CLINICAL_REMAINING_COLS = ['PATIENT_ID', 'SAMPLE_ID', 'YEAR_OF_DIAGNOSIS', 'os_label']

ORDINAL_CATEGORICAL_COLS = ['PATH_T_STAGE', 'PATH_N_STAGE', 'PATH_M_STAGE', 'PATH_STAGE']
NOMINAL_CATEGORICAL_COLS = [c for c in CLINICAL_CATEGORICAL_COLS if c not in ORDINAL_CATEGORICAL_COLS]

# X/NOS = not assessed / not otherwise specified, placed last
T_STAGE_ORDER = ['Tis', 'T0', 'T1', 'T1a', 'T1b', 'T2', 'T2a', 'T2b', 'T3', 'T3a', 'T3b', 'T4', 'T4a', 'T4b', 'TX']
N_STAGE_ORDER = ['N0', 'N1', 'N1a', 'N1b', 'N2', 'N2a', 'N2b', 'N2c', 'N3', 'NX']
M_STAGE_ORDER = ['M0', 'M1a', 'M1b', 'M1c', 'M1']
PATH_STAGE_ORDER = ['Stage 0', 'Stage I', 'Stage IA', 'Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB',
                     'Stage IIC', 'Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC', 'Stage IV', 'I/II NOS']

PARAMS = {
    "k": 10,
    "variance_threshold": 0.1,
    "random_state":42,
    "test_size": 0.2,
    "optuna_trials": 10
}

# Load the clinical+gex corrected dataset
clinical_gex = pd.read_csv(DATA_PATH, low_memory=False)
clinical_gex.columns = clinical_gex.columns.str.strip()

gex_cols = [c for c in clinical_gex.columns if c not in CLINICAL_CATEGORICAL_COLS
            + CLINICAL_NUMERIC_COLS + CLINICAL_REMAINING_COLS]

x = clinical_gex[CLINICAL_CATEGORICAL_COLS + CLINICAL_NUMERIC_COLS + gex_cols]
y = clinical_gex['os_label']

# Split first, so gene selection below only ever sees the training split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=PARAMS['test_size'], random_state=PARAMS['random_state'], stratify=y
)

# Build the preprocessing + logistic regression pipeline
preprocess_features = ColumnTransformer(
    transformers=[
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='if_binary')),
        ]), NOMINAL_CATEGORICAL_COLS),

        ('ordinal', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ord', OrdinalEncoder(categories=[T_STAGE_ORDER, N_STAGE_ORDER, M_STAGE_ORDER, PATH_STAGE_ORDER],
                                    handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), ORDINAL_CATEGORICAL_COLS),

        ('numeric', SimpleImputer(strategy='median'), CLINICAL_NUMERIC_COLS),
        ('gex_selection', Pipeline([
            ('threshold', VarianceThreshold(threshold=PARAMS['variance_threshold'])),
            ('selectkbest', SelectKBest(f_classif, k=PARAMS['k'])),
        ]), gex_cols),
    ],
    remainder='drop'
)

prep = Pipeline([
    ('preprocess', preprocess_features),
    ('scaler', StandardScaler()),
])

x_train_tf = prep.fit_transform(x_train, y_train)
x_test_tf = prep.transform(x_test)

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f'trial_{trial.number}') as child_run:
        
        lr_params = {
            'C':        trial.suggest_float('clf__C', 1e-3, 1.0, log=True),
            'l1_ratio': trial.suggest_float('clf__l1_ratio', 0.0, 1.0),
        }
        
        clf = LogisticRegression(
            solver="saga", 
            class_weight="balanced", 
            max_iter=5000,
            random_state=PARAMS['random_state'],
            C=lr_params['C'],
            l1_ratio=lr_params['l1_ratio']
        )

        mlflow.log_params(lr_params)
        
        clf.fit(x_train_tf, y_train)
        predictions = clf.predict(x_test_tf)
        
        f1 = metrics.f1_score(y_test, predictions)
        acc = metrics.accuracy_score(y_test, predictions)
        
        print(f'F1 = {f1}\nAcc = {acc}')
        
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(
            clf,
            name='VT_SKB_LR',
            input_example=x_train_tf[:5],
            skops_trusted_types=["numpy.dtype", "sklearn.feature_selection._univariate_selection.f_classif"],
        )

        selected_gex = preprocess_features.named_transformers_['gex_selection'].get_feature_names_out()
        mlflow.log_text("\n".join(selected_gex), "selected_genes.txt")
        
        # retrieve best-performing child run later
        trial.set_user_attr('run_id', child_run.info.run_id)
        
        return acc
    
with mlflow.start_run(run_name='study') as run:
    n_trials = PARAMS['optuna_trials']
    mlflow.log_params(PARAMS)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Log the best trial and its run ID
    mlflow.log_params(study.best_trial.params)
    mlflow.log_metrics({
        'best_error': study.best_value
    })
    if best_run_id := study.best_trial.user_attrs.get('run_id'):
        mlflow.log_param('best_child_run_id', best_run_id)