import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

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

RANDOM_STATE = 42
K = 20

# Load the clinical+gex corrected dataset
clinical_gex = pd.read_csv(DATA_PATH, low_memory=False)
clinical_gex.columns = clinical_gex.columns.str.strip()
# print(clinical_gex.shape)

gex_cols = [c for c in clinical_gex.columns if c not in CLINICAL_CATEGORICAL_COLS
            + CLINICAL_NUMERIC_COLS + CLINICAL_REMAINING_COLS]

x = clinical_gex[CLINICAL_CATEGORICAL_COLS + CLINICAL_NUMERIC_COLS + gex_cols]
y = clinical_gex['os_label']

# Split first, so gene selection below only ever sees the training split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Select the top-k gex features by ANOVA F-value, fit on the training split only to avoid data leakage
pre_selection_pipe = Pipeline([
    ('threshold', VarianceThreshold(threshold=0.1)),
    ('selectkbest', SelectKBest(f_classif, k=K)),
])
pre_selection_pipe.fit(x_train[gex_cols], y_train)
gex_cols_selected = list(pre_selection_pipe.get_feature_names_out())

features_after_vt = pre_selection_pipe['threshold'].get_feature_names_out()
f_values = pre_selection_pipe['selectkbest'].scores_
f_series = pd.Series(f_values, index=features_after_vt).sort_values(ascending=False)
# print("Top 20 genes by ANOVA F-value:")
# print(f_series.head(20).to_string())

feature_cols = CLINICAL_CATEGORICAL_COLS + CLINICAL_NUMERIC_COLS + gex_cols_selected
x_train = x_train[feature_cols]
x_test = x_test[feature_cols]

# print(f'Dropped {len(CLINICAL_REMAINING_COLS)} clinical cols, {x_train.shape} remaining')
# print(f"\nTarget distribution:\n{y.value_counts().sort_index()}")

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
    ],
    remainder='passthrough',
)

lr_clf = LogisticRegression(
    solver='saga',
    class_weight='balanced',
    max_iter=5000,
    random_state=RANDOM_STATE,
)

full_pipe = Pipeline([
    ('preprocess', preprocess_features),
    ('scaler', StandardScaler()),
    ('clf', lr_clf),
])

full_pipe.fit(x_train, y_train)

# Evaluate on the validation split
predictions = full_pipe.predict(x_test)
print(f"F1: {metrics.f1_score(y_test, predictions)}")
print(f"Accuracy: {metrics.accuracy_score(y_test, predictions)}")
