from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. OPTIONAL MODEL IMPORTS
# ============================================================

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "processed"

OUTPUT_DIR = (
    PROCESSED_DIR
    / "ml_modeling_outputs"
)

FIGURE_DIR = (
    OUTPUT_DIR
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 3. CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 3

TUNING_SAMPLE_SIZE = 150_000

N_ITER_SEARCH = 6

N_JOBS = 2

TARGET = "total_cost_INR"

ENGINEERING_GROUP_FEATURES = [
    "ADL_kNm",
    "LL_kNm",
    "span_cc_mm",
    "fck_MPa",
    "fy_MPa"
]

MODEL_FEATURES = [
    "ADL_kNm",
    "LL_kNm",
    "span_cc_mm",
    "b_mm",
    "D_final_mm",
    "d_final_mm",
    "fck_MPa",
    "fy_MPa",
    "phi_main_mm",
    "phi_stirrup_mm",
    "cover_mm",
    "Ast_prov_mm2",
    "n_bars",
    "bar_spacing_mm",
    "pt_prov_percent",
    "leff_mm"
]


# ============================================================
# 4. FILE PATHS
# ============================================================

TRAIN_PATH = (
    PROCESSED_DIR
    / "train.csv"
)

VALIDATION_PATH = (
    PROCESSED_DIR
    / "validation.csv"
)

TEST_PATH = (
    PROCESSED_DIR
    / "test.csv"
)


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def calculate_mape(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.maximum(
        np.abs(y_true),
        1e-8
    )

    return np.mean(
        np.abs(
            (y_true - y_pred)
            / denominator
        )
    ) * 100.0


def evaluate_predictions(
    y_true,
    y_pred
):

    return {
        "MAE_INR":
            mean_absolute_error(
                y_true,
                y_pred
            ),

        "RMSE_INR":
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            ),

        "R2":
            r2_score(
                y_true,
                y_pred
            ),

        "MAPE_percent":
            calculate_mape(
                y_true,
                y_pred
            )
    }


def make_engineering_groups(df):

    return (
        df[
            ENGINEERING_GROUP_FEATURES
        ]
        .astype(str)
        .agg(
            "_".join,
            axis=1
        )
    )


def print_metrics(
    label,
    metrics
):

    print()
    print(label)
    print(
        f"MAE  : ₹{metrics['MAE_INR']:,.2f}"
    )
    print(
        f"RMSE : ₹{metrics['RMSE_INR']:,.2f}"
    )
    print(
        f"R²   : {metrics['R2']:.6f}"
    )
    print(
        f"MAPE : {metrics['MAPE_percent']:.3f}%"
    )


# ============================================================
# 6. LOAD DATA
# ============================================================

print("=" * 75)
print("RCC BEAM ML MODELING")
print("=" * 75)
print()

print("Loading datasets...")

train_df = pd.read_csv(
    TRAIN_PATH
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

test_df = pd.read_csv(
    TEST_PATH
)

print(
    f"Training rows   : "
    f"{len(train_df):,}"
)

print(
    f"Validation rows : "
    f"{len(validation_df):,}"
)

print(
    f"Test rows       : "
    f"{len(test_df):,}"
)


# ============================================================
# 7. COLUMN VALIDATION
# ============================================================

required_columns = (
    MODEL_FEATURES
    + [TARGET]
    + ENGINEERING_GROUP_FEATURES
)

for name, df in [
    ("train", train_df),
    ("validation", validation_df),
    ("test", test_df)
]:

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{name}.csv is missing columns: "
            f"{missing}"
        )


# ============================================================
# 8. PREPARE X / y
# ============================================================

X_train = train_df[
    MODEL_FEATURES
].copy()

y_train = train_df[
    TARGET
].copy()

X_validation = validation_df[
    MODEL_FEATURES
].copy()

y_validation = validation_df[
    TARGET
].copy()

X_test = test_df[
    MODEL_FEATURES
].copy()

y_test = test_df[
    TARGET
].copy()


# ============================================================
# 9. ENGINEERING GROUPS
# ============================================================

groups_train = make_engineering_groups(
    train_df
)

groups_validation = make_engineering_groups(
    validation_df
)

groups_test = make_engineering_groups(
    test_df
)

print()
print("Engineering groups:")

print(
    f"Train      : "
    f"{groups_train.nunique():,}"
)

print(
    f"Validation : "
    f"{groups_validation.nunique():,}"
)

print(
    f"Test       : "
    f"{groups_test.nunique():,}"
)


# ============================================================
# 10. CHECK GROUP OVERLAP
# ============================================================

train_groups_set = set(
    groups_train.unique()
)

validation_groups_set = set(
    groups_validation.unique()
)

test_groups_set = set(
    groups_test.unique()
)

train_validation_overlap = (
    train_groups_set
    & validation_groups_set
)

train_test_overlap = (
    train_groups_set
    & test_groups_set
)

validation_test_overlap = (
    validation_groups_set
    & test_groups_set
)

if train_validation_overlap:
    raise ValueError(
        "Engineering-group overlap between "
        "training and validation sets."
    )

if train_test_overlap:
    raise ValueError(
        "Engineering-group overlap between "
        "training and test sets."
    )

if validation_test_overlap:
    raise ValueError(
        "Engineering-group overlap between "
        "validation and test sets."
    )

print()
print(
    "Engineering-group overlap: ZERO"
)


# ============================================================
# 11. GROUPED CV TUNING SAMPLE
# ============================================================

if len(train_df) > TUNING_SAMPLE_SIZE:

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    tuning_indices = (
        rng.choice(
            len(train_df),
            size=TUNING_SAMPLE_SIZE,
            replace=False
        )
    )

    tuning_indices = np.sort(
        tuning_indices
    )

else:

    tuning_indices = np.arange(
        len(train_df)
    )


X_tuning = X_train.iloc[
    tuning_indices
].copy()

y_tuning = y_train.iloc[
    tuning_indices
].copy()

groups_tuning = groups_train.iloc[
    tuning_indices
].copy()


print()
print(
    f"Tuning rows    : "
    f"{len(X_tuning):,}"
)

print(
    f"Tuning groups  : "
    f"{groups_tuning.nunique():,}"
)


# ============================================================
# 12. GROUP K-FOLD
# ============================================================

group_cv = GroupKFold(
    n_splits=N_SPLITS
)


# ============================================================
# 13. MODEL DEFINITIONS
# ============================================================

models = {}

parameter_distributions = {}


# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

models["Random Forest"] = (
    RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
)

parameter_distributions[
    "Random Forest"
] = {
    "n_estimators": [
        150,
        200,
        300
    ],

    "max_depth": [
        20,
        30,
        40,
        None
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "min_samples_split": [
        2,
        5
    ],

    "max_features": [
        0.6,
        0.8,
        1.0
    ],

    "bootstrap": [
        True
    ]
}


# ------------------------------------------------------------
# Extra Trees
# ------------------------------------------------------------

models["Extra Trees"] = (
    ExtraTreesRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
)

parameter_distributions[
    "Extra Trees"
] = {
    "n_estimators": [
        150,
        200,
        300
    ],

    "max_depth": [
        20,
        30,
        40,
        None
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "min_samples_split": [
        2,
        5
    ],

    "max_features": [
        0.6,
        0.8,
        1.0
    ],

    "bootstrap": [
        False
    ]
}


# ------------------------------------------------------------
# HistGradientBoosting
# ------------------------------------------------------------

models[
    "HistGradientBoosting"
] = (
    HistGradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
)

parameter_distributions[
    "HistGradientBoosting"
] = {
    "max_iter": [
        300,
        500,
        700
    ],

    "learning_rate": [
        0.05,
        0.08,
        0.10
    ],

    "max_leaf_nodes": [
        15,
        31,
        63
    ],

    "max_depth": [
        None,
        6,
        8
    ],

    "min_samples_leaf": [
        10,
        20,
        30
    ],

    "l2_regularization": [
        0,
        1,
        5
    ]
}


# ------------------------------------------------------------
# Gradient Boosting
# ------------------------------------------------------------

models[
    "Gradient Boosting"
] = (
    GradientBoostingRegressor(
        random_state=RANDOM_STATE
    )
)

parameter_distributions[
    "Gradient Boosting"
] = {
    "n_estimators": [
        200,
        400,
        600
    ],

    "learning_rate": [
        0.03,
        0.05,
        0.08,
        0.10
    ],

    "max_depth": [
        2,
        3,
        4,
        5
    ],

    "min_samples_leaf": [
        2,
        5,
        10
    ],

    "subsample": [
        0.8,
        1.0
    ]
}


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------

if XGBOOST_AVAILABLE:

    models[
        "XGBoost"
    ] = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        tree_method="hist"
    )

    parameter_distributions[
        "XGBoost"
    ] = {
        "n_estimators": [
            500,
            700,
            1000
        ],

        "max_depth": [
            6,
            8,
            10
        ],

        "learning_rate": [
            0.05,
            0.08,
            0.10
        ],

        "min_child_weight": [
            1,
            3,
            5
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ],

        "reg_alpha": [
            0,
            0.1
        ],

        "reg_lambda": [
            1,
            10
        ],

        "gamma": [
            0,
            0.1
        ]
    }

else:

    print()
    print(
        "WARNING: XGBoost is not installed."
    )


# ------------------------------------------------------------
# CatBoost
# ------------------------------------------------------------

if CATBOOST_AVAILABLE:

    models[
        "CatBoost"
    ] = CatBoostRegressor(
        loss_function="MAE",
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=N_JOBS
    )

    parameter_distributions[
        "CatBoost"
    ] = {
        "iterations": [
            500,
            700,
            1000
        ],

        "depth": [
            6,
            8,
            10
        ],

        "learning_rate": [
            0.05,
            0.08,
            0.10
        ],

        "l2_leaf_reg": [
            3,
            5,
            10
        ],

        "random_strength": [
            0,
            1
        ]
    }

else:

    print()
    print(
        "WARNING: CatBoost is not installed."
    )


# ------------------------------------------------------------
# LightGBM
# ------------------------------------------------------------

if LIGHTGBM_AVAILABLE:

    models[
        "LightGBM"
    ] = LGBMRegressor(
        objective="regression",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1
    )

    parameter_distributions[
        "LightGBM"
    ] = {
        "n_estimators": [
            300,
            500,
            700
        ],

        "learning_rate": [
            0.05,
            0.08,
            0.10
        ],

        "num_leaves": [
            31,
            63,
            127
        ],

        "max_depth": [
            -1,
            8,
            12
        ],

        "min_child_samples": [
            20,
            50,
            100
        ],

        "subsample": [
            0.8,
            1.0
        ],

        "colsample_bytree": [
            0.8,
            1.0
        ],

        "reg_alpha": [
            0,
            0.1
        ],

        "reg_lambda": [
            0,
            1,
            10
        ]
    }

else:

    print()
    print(
        "WARNING: LightGBM is not installed."
    )


# ============================================================
# 14. MODEL TUNING
# ============================================================

all_results = []

best_estimators = {}

best_parameters = {}

group_cv_scores = {}


for model_name, model in models.items():

    print()
    print("=" * 75)
    print(
        f"MODEL: {model_name}"
    )
    print("=" * 75)

    start_time = time.time()

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=(
            parameter_distributions[
                model_name
            ]
        ),
        n_iter=N_ITER_SEARCH,
        scoring="neg_mean_absolute_error",
        cv=group_cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=False
    )

    search.fit(
        X_tuning,
        y_tuning,
        groups=groups_tuning
    )

    tuning_time = (
        time.time()
        - start_time
    )

    best_cv_mae = (
        -search.best_score_
    )

    best_model = search.best_estimator_

    best_estimators[
        model_name
    ] = best_model

    best_parameters[
        model_name
    ] = search.best_params_

    group_cv_scores[
        model_name
    ] = best_cv_mae

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    validation_start = time.time()

    validation_prediction = (
        best_model.predict(
            X_validation
        )
    )

    validation_time = (
        time.time()
        - validation_start
    )

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_prediction
    )

    result = {
        "model":
            model_name,

        "group_cv_mae_INR":
            best_cv_mae,

        "validation_mae_INR":
            validation_metrics[
                "MAE_INR"
            ],

        "validation_rmse_INR":
            validation_metrics[
                "RMSE_INR"
            ],

        "validation_r2":
            validation_metrics[
                "R2"
            ],

        "validation_mape_percent":
            validation_metrics[
                "MAPE_percent"
            ],

        "tuning_time_seconds":
            tuning_time,

        "validation_prediction_time_seconds":
            validation_time
    }

    all_results.append(
        result
    )

    print()
    print(
        f"Best Group-CV MAE:"
    )
    print(
        f"₹{best_cv_mae:,.2f}"
    )

    print_metrics(
        "Validation:",
        validation_metrics
    )

    print()
    print(
        "Best parameters:"
    )

    print(
        search.best_params_
    )

    print()
    print(
        f"Tuning time: "
        f"{tuning_time:.2f} seconds"
    )


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    all_results
)

comparison_df = comparison_df.sort_values(
    by="validation_mae_INR",
    ascending=True
).reset_index(drop=True)


comparison_path = (
    OUTPUT_DIR
    / "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)


print()
print("=" * 75)
print("GROUPED MODEL COMPARISON")
print("=" * 75)

print(
    comparison_df[
        [
            "model",
            "group_cv_mae_INR",
            "validation_mae_INR",
            "validation_rmse_INR",
            "validation_r2",
            "validation_mape_percent",
            "tuning_time_seconds"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

best_model_name = (
    comparison_df.iloc[0]["model"]
)

best_model_validation_mae = (
    comparison_df.iloc[0][
        "validation_mae_INR"
    ]
)

selected_model = best_estimators[
    best_model_name
]


print()
print("=" * 75)
print("BEST MODEL")
print("=" * 75)

print(
    best_model_name
)

print(
    f"Validation MAE: "
    f"₹{best_model_validation_mae:,.2f}"
)


# ============================================================
# 17. SAVE BEST MODEL FROM TUNING
# ============================================================

tuned_model_path = (
    OUTPUT_DIR
    / "best_tuned_model.joblib"
)

joblib.dump(
    selected_model,
    tuned_model_path
)


# ============================================================
# 18. VALIDATION PREDICTION PLOT
# ============================================================

validation_prediction = (
    selected_model.predict(
        X_validation
    )
)

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_validation,
    validation_prediction,
    s=8,
    alpha=0.35
)

minimum_value = min(
    y_validation.min(),
    validation_prediction.min()
)

maximum_value = max(
    y_validation.max(),
    validation_prediction.max()
)

plt.plot(
    [
        minimum_value,
        maximum_value
    ],
    [
        minimum_value,
        maximum_value
    ],
    linestyle="--"
)

plt.xlabel(
    "Actual Cost (INR)"
)

plt.ylabel(
    "Predicted Cost (INR)"
)

plt.title(
    f"{best_model_name} — Validation: "
    "Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "validation_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. VALIDATION RESIDUAL PLOT
# ============================================================

validation_residuals = (
    y_validation.to_numpy()
    - validation_prediction
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    validation_prediction,
    validation_residuals,
    s=8,
    alpha=0.35
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Cost (INR)"
)

plt.ylabel(
    "Residual = Actual − Predicted (INR)"
)

plt.title(
    f"{best_model_name} — Validation Residuals"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "validation_residuals.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. FINAL RETRAINING
# ============================================================

print()
print("=" * 75)
print("FINAL RETRAINING")
print("=" * 75)


X_train_final = pd.concat(
    [
        X_train,
        X_validation
    ],
    axis=0
).reset_index(
    drop=True
)

y_train_final = pd.concat(
    [
        y_train,
        y_validation
    ],
    axis=0
).reset_index(
    drop=True
)


print(
    f"Final training rows: "
    f"{len(X_train_final):,}"
)


# ============================================================
# 21. FINAL MODEL
# ============================================================

final_model = clone(
    selected_model
)

final_start_time = time.time()

final_model.fit(
    X_train_final,
    y_train_final
)

final_training_time = (
    time.time()
    - final_start_time
)

print(
    f"Final training time: "
    f"{final_training_time:.2f} seconds"
)


# ============================================================
# 22. FINAL TEST EVALUATION
# ============================================================

print()
print("=" * 75)
print("FINAL UNSEEN ENGINEERING-PROBLEM TEST")
print("=" * 75)


test_start_time = time.time()

test_prediction = (
    final_model.predict(
        X_test
    )
)

test_prediction_time = (
    time.time()
    - test_start_time
)

test_metrics = evaluate_predictions(
    y_test,
    test_prediction
)

print_metrics(
    "Test:",
    test_metrics
)

print(
    f"Prediction time: "
    f"{test_prediction_time:.4f} seconds"
)


# ============================================================
# 23. SAVE FINAL MODEL
# ============================================================

final_model_path = (
    OUTPUT_DIR
    / "final_ml_model.joblib"
)

joblib.dump(
    final_model,
    final_model_path
)


# ============================================================
# 24. SAVE TEST PREDICTIONS
# ============================================================

test_predictions_df = pd.DataFrame({
    "actual_cost_INR":
        y_test.to_numpy(),

    "predicted_cost_INR":
        test_prediction
})

test_predictions_df[
    "residual_INR"
] = (
    test_predictions_df[
        "actual_cost_INR"
    ]
    -
    test_predictions_df[
        "predicted_cost_INR"
    ]
)

test_predictions_path = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

test_predictions_df.to_csv(
    test_predictions_path,
    index=False
)


# ============================================================
# 25. FINAL ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(
    figsize=(8, 7)
)

plt.scatter(
    y_test,
    test_prediction,
    s=8,
    alpha=0.35
)

minimum_value = min(
    y_test.min(),
    test_prediction.min()
)

maximum_value = max(
    y_test.max(),
    test_prediction.max()
)

plt.plot(
    [
        minimum_value,
        maximum_value
    ],
    [
        minimum_value,
        maximum_value
    ],
    linestyle="--"
)

plt.xlabel(
    "Actual Cost (INR)"
)

plt.ylabel(
    "Predicted Cost (INR)"
)

plt.title(
    f"{best_model_name} — Unseen Test: "
    "Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "test_actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ============================================================
# 26. FINAL RESIDUAL PLOT
# ============================================================

test_residuals = (
    y_test.to_numpy()
    - test_prediction
)

plt.figure(
    figsize=(9, 6)
)

plt.scatter(
    test_prediction,
    test_residuals,
    s=8,
    alpha=0.35
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Predicted Cost (INR)"
)

plt.ylabel(
    "Residual = Actual − Predicted (INR)"
)

plt.title(
    f"{best_model_name} — Unseen Test Residuals"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "test_residuals.png",
    dpi=300
)

plt.close()


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

if hasattr(
    final_model,
    "feature_importances_"
):

    feature_importance = (
        np.asarray(
            final_model.feature_importances_
        )
    )

elif hasattr(
    final_model,
    "get_feature_importance"
):

    feature_importance = (
        np.asarray(
            final_model.get_feature_importance()
        )
    )

else:

    feature_importance = None


if feature_importance is not None:

    feature_importance_df = pd.DataFrame({
        "feature":
            MODEL_FEATURES,

        "importance":
            feature_importance
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by="importance",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    feature_importance_path = (
        OUTPUT_DIR
        / "feature_importance.csv"
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False
    )

    print()
    print("=" * 75)
    print("FEATURE IMPORTANCE")
    print("=" * 75)

    print(
        feature_importance_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature importance plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    plot_df = (
        feature_importance_df
        .sort_values(
            by="importance",
            ascending=True
        )
    )

    plt.barh(
        plot_df["feature"],
        plot_df["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        f"{best_model_name} — Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "feature_importance.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 28. FINAL METRICS FILE
# ============================================================

final_metrics = {
    "best_model":
        best_model_name,

    "validation_mae_INR":
        float(
            best_model_validation_mae
        ),

    "test_mae_INR":
        float(
            test_metrics["MAE_INR"]
        ),

    "test_rmse_INR":
        float(
            test_metrics["RMSE_INR"]
        ),

    "test_r2":
        float(
            test_metrics["R2"]
        ),

    "test_mape_percent":
        float(
            test_metrics["MAPE_percent"]
        ),

    "final_training_rows":
        int(
            len(X_train_final)
        ),

    "final_training_time_seconds":
        float(
            final_training_time
        ),

    "test_prediction_time_seconds":
        float(
            test_prediction_time
        )
}


metrics_path = (
    OUTPUT_DIR
    / "final_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_metrics,
        f,
        indent=4
    )


# ============================================================
# 29. SAVE BEST PARAMETERS
# ============================================================

parameters_serializable = {}

for model_name, params in best_parameters.items():

    parameters_serializable[
        model_name
    ] = {
        key: (
            value.item()
            if isinstance(
                value,
                np.generic
            )
            else value
        )
        for key, value in params.items()
    }


parameters_path = (
    OUTPUT_DIR
    / "best_parameters.json"
)

with open(
    parameters_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        parameters_serializable,
        f,
        indent=4
    )


# ============================================================
# 30. MODEL COMPARISON FIGURE
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    comparison_df["model"],
    comparison_df[
        "validation_mae_INR"
    ]
)

plt.ylabel(
    "Validation MAE (INR)"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Grouped Validation MAE — ML Model Comparison"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "model_comparison_mae.png",
    dpi=300
)

plt.close()


# ============================================================
# 31. FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("FINAL ML MODELING RESULTS")
print("=" * 75)

print(
    f"Best model          : "
    f"{best_model_name}"
)

print(
    f"Validation MAE      : "
    f"₹{best_model_validation_mae:,.2f}"
)

print(
    f"Unseen test MAE     : "
    f"₹{test_metrics['MAE_INR']:,.2f}"
)

print(
    f"Unseen test RMSE    : "
    f"₹{test_metrics['RMSE_INR']:,.2f}"
)

print(
    f"Unseen test R²      : "
    f"{test_metrics['R2']:.6f}"
)

print(
    f"Unseen test MAPE    : "
    f"{test_metrics['MAPE_percent']:.3f}%"
)

print()

print("=" * 75)
print("OUTPUT FILES")
print("=" * 75)

print(
    f"Model comparison    : "
    f"{comparison_path}"
)

print(
    f"Best tuned model    : "
    f"{tuned_model_path}"
)

print(
    f"Final ML model      : "
    f"{final_model_path}"
)

print(
    f"Test predictions    : "
    f"{test_predictions_path}"
)

print(
    f"Final metrics       : "
    f"{metrics_path}"
)

print(
    f"Best parameters     : "
    f"{parameters_path}"
)

print(
    f"Figures             : "
    f"{FIGURE_DIR}"
)

print()
print(
    "ML modeling completed successfully."
)