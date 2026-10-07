from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"

OUTPUT_DIR = PROCESSED_DIR / "ml_optimization_outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

CANDIDATE_FILE = PROCESSED_DIR / "feasible_candidates.csv"
MODEL_FILE = (
    PROCESSED_DIR
    / "ml_modeling_outputs"
    / "final_ml_model.joblib"
)

TARGET = "total_cost_INR"

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
    "leff_mm",
]

TOP_K_VALUES = [100, 500, 1000]


def calculate_bar_spacing(row):
    b = float(row["b_mm"])
    cover = float(row["cover_mm"])
    phi_stirrup = float(row["phi_stirrup_mm"])
    phi_main = float(row["phi_main_mm"])
    n_bars = int(row["n_bars"])

    if n_bars <= 1:
        return np.inf

    numerator = (
        b
        - 2.0 * cover
        - 2.0 * phi_stirrup
        - n_bars * phi_main
    )

    denominator = n_bars - 1

    return numerator / denominator


def load_inputs():

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{CANDIDATE_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"ML model not found:\n{MODEL_FILE}\n"
            "Run ml_modeling.py first."
        )

    candidates = pd.read_csv(CANDIDATE_FILE)
    model = joblib.load(MODEL_FILE)

    return candidates, model


def validate_columns(df):

    required = [
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
        "pt_prov_percent",
        "leff_mm",
        TARGET,
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing)
        )


def prepare_features(df):

    df = df.copy()

    df["bar_spacing_mm"] = df.apply(
        calculate_bar_spacing,
        axis=1
    )

    valid_spacing = (
        np.isfinite(df["bar_spacing_mm"])
        & (df["bar_spacing_mm"] > 0)
    )

    df = df.loc[valid_spacing].copy()

    df["pt_prov_percent"] = (
        100.0
        * df["Ast_prov_mm2"]
        / (
            df["b_mm"]
            * df["d_final_mm"]
        )
    )

    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=MODEL_FEATURES + [TARGET]
    ).copy()

    return df


def predict_cost(df, model):

    X = df[MODEL_FEATURES].copy()

    predictions = model.predict(X)

    df["ml_predicted_cost_INR"] = predictions

    df["absolute_prediction_error_INR"] = (
        df[TARGET]
        - df["ml_predicted_cost_INR"]
    ).abs()

    df["prediction_error_percent"] = (
        df["absolute_prediction_error_INR"]
        / df[TARGET]
        * 100.0
    )

    return df


def rank_candidates(df):

    ranked = df.sort_values(
        "ml_predicted_cost_INR",
        ascending=True
    ).reset_index(drop=True)

    ranked["ML_rank"] = (
        np.arange(len(ranked)) + 1
    )

    return ranked


def evaluate_top_k(ranked, k):

    top_k = ranked.head(k).copy()

    best_idx = top_k[TARGET].idxmin()

    best = top_k.loc[best_idx].copy()

    return best


def find_actual_best(df):

    idx = df[TARGET].idxmin()

    return df.loc[idx].copy()


def calculate_regret(
    actual_best_cost,
    selected_cost
):

    difference = (
        selected_cost
        - actual_best_cost
    )

    percentage = (
        difference
        / actual_best_cost
        * 100.0
    )

    return difference, percentage


def print_design(title, row):

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)

    fields = [
        ("Width b", "b_mm", "mm"),
        ("Overall depth D", "D_final_mm", "mm"),
        ("Effective depth d", "d_final_mm", "mm"),
        ("Main bar diameter", "phi_main_mm", "mm"),
        ("Number of main bars", "n_bars", ""),
        ("Main bar spacing", "bar_spacing_mm", "mm"),
        ("Stirrup diameter", "phi_stirrup_mm", "mm"),
        ("Stirrup spacing", "stirrup_spacing_mm", "mm"),
        ("Ast provided", "Ast_prov_mm2", "mm²"),
        ("pt provided", "pt_prov_percent", "%"),
        ("Effective span", "leff_mm", "mm"),
        ("Actual cost", TARGET, "₹"),
        ("ML predicted cost", "ml_predicted_cost_INR", "₹"),
        ("ML rank", "ML_rank", ""),
    ]

    for label, column, unit in fields:

        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        if column in [
            "b_mm",
            "D_final_mm",
            "d_final_mm",
            "phi_main_mm",
            "n_bars",
            "bar_spacing_mm",
            "phi_stirrup_mm",
            "stirrup_spacing_mm",
            "Ast_prov_mm2",
            "leff_mm",
        ]:

            if isinstance(
                value,
                (int, np.integer)
            ):
                formatted = f"{int(value)}"
            else:
                formatted = f"{float(value):,.2f}"

        elif column == "pt_prov_percent":

            formatted = f"{float(value):.2f}"

        elif column in [
            TARGET,
            "ml_predicted_cost_INR",
        ]:

            formatted = f"{float(value):,.2f}"

        elif column == "ML_rank":

            formatted = f"{int(value)}"

        else:

            formatted = str(value)

        print(
            f"{label:<28}: "
            f"{formatted} {unit}"
        )


def save_outputs(
    ranked,
    actual_best,
    top_results,
    summary
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ranked.to_csv(
        OUTPUT_DIR
        / "ranked_candidates.csv",
        index=False
    )

    ranked.to_csv(
        OUTPUT_DIR
        / "all_candidate_predictions.csv",
        index=False
    )

    ranked.head(50).to_csv(
        OUTPUT_DIR
        / "top_50_ml_designs.csv",
        index=False
    )

    ranked.head(500).to_csv(
        OUTPUT_DIR
        / "top_500_ml_candidates.csv",
        index=False
    )

    pd.DataFrame(
        top_results
    ).drop(
        columns=["selected_design"],
        errors="ignore"
    ).to_csv(
        OUTPUT_DIR
        / "top_k_optimization_results.csv",
        index=False
    )

    pd.DataFrame(
        [actual_best]
    ).to_csv(
        OUTPUT_DIR
        / "actual_best_design.csv",
        index=False
    )

    selected_final = (
        top_results[-1]["selected_design"]
    )

    pd.DataFrame(
        [selected_final]
    ).to_csv(
        OUTPUT_DIR
        / "ml_optimal_design.csv",
        index=False
    )

    comparison_rows = []

    for result in top_results:

        comparison_rows.append({
            "top_k":
                result["top_k"],

            "selected_actual_cost_INR":
                result[
                    "selected_actual_cost_INR"
                ],

            "actual_best_cost_INR":
                result[
                    "actual_best_cost_INR"
                ],

            "difference_INR":
                result["difference_INR"],

            "difference_percent":
                result["difference_percent"],

            "search_reduction_percent":
                result[
                    "search_reduction_percent"
                ],

            "actual_best_recovered":
                result[
                    "actual_best_recovered"
                ],

            "ml_rank_of_selected_design":
                result[
                    "ml_rank_of_selected_design"
                ],
        })

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        OUTPUT_DIR
        / "optimization_comparison.csv",
        index=False
    )

    with open(
        OUTPUT_DIR
        / "optimization_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    create_figures(
        ranked,
        actual_best,
        top_results
    )


def create_figures(
    ranked,
    actual_best,
    top_results
):

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        ranked[TARGET],
        ranked["ml_predicted_cost_INR"],
        s=8,
        alpha=0.25
    )

    minimum = min(
        ranked[TARGET].min(),
        ranked[
            "ml_predicted_cost_INR"
        ].min()
    )

    maximum = max(
        ranked[TARGET].max(),
        ranked[
            "ml_predicted_cost_INR"
        ].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel(
        "Actual Cost (INR)"
    )

    plt.ylabel(
        "ML Predicted Cost (INR)"
    )

    plt.title(
        "Actual vs ML-Predicted Cost"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "actual_vs_ml_cost.png",
        dpi=300
    )

    plt.close()

    top_k = [
        str(result["top_k"])
        for result in top_results
    ]

    selected_costs = [
        result[
            "selected_actual_cost_INR"
        ]
        for result in top_results
    ]

    actual_cost = float(
        actual_best[TARGET]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        top_k,
        selected_costs,
        marker="o",
        label="Best actual cost found"
    )

    plt.axhline(
        actual_cost,
        linestyle="--",
        label="Exhaustive optimum"
    )

    plt.xlabel(
        "Number of ML-screened candidates"
    )

    plt.ylabel(
        "Best Actual Cost (INR)"
    )

    plt.title(
        "ML-Assisted Optimization"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "ml_screening_vs_exhaustive.png",
        dpi=300
    )

    plt.close()

    top20 = ranked.head(20).copy()

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        np.arange(len(top20)),
        top20[
            "ml_predicted_cost_INR"
        ]
    )

    plt.xlabel(
        "ML Candidate Rank"
    )

    plt.ylabel(
        "Predicted Cost (INR)"
    )

    plt.title(
        "Top 20 ML-Ranked Designs"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "top_20_ml_designs.png",
        dpi=300
    )

    plt.close()

    plt.figure(
        figsize=(8, 6)
    )

    ranked_1000 = ranked.head(1000)

    plt.plot(
        ranked_1000["ML_rank"],
        ranked_1000[TARGET],
        linewidth=1
    )

    plt.axhline(
        actual_cost,
        linestyle="--",
        label="Exhaustive optimum"
    )

    plt.xlabel(
        "ML Rank"
    )

    plt.ylabel(
        "Actual Cost (INR)"
    )

    plt.title(
        "Actual Cost Across ML Ranking"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "actual_cost_vs_ml_rank.png",
        dpi=300
    )

    plt.close()


def main():

    print("=" * 70)
    print("ML-ASSISTED RCC BEAM OPTIMIZATION")
    print("=" * 70)

    print()

    print(
        "Candidate file :",
        CANDIDATE_FILE
    )

    print(
        "Model file     :",
        MODEL_FILE
    )

    candidates, model = load_inputs()

    print(
        "Candidates     :",
        f"{len(candidates):,}"
    )

    validate_columns(
        candidates
    )

    candidates = prepare_features(
        candidates
    )

    print(
        "Valid candidates:",
        f"{len(candidates):,}"
    )

    print()
    print(
        "Generating ML cost predictions..."
    )
    print()

    candidates = predict_cost(
        candidates,
        model
    )

    ranked = rank_candidates(
        candidates
    )

    actual_best = find_actual_best(
        ranked
    )

    print_design(
        "EXHAUSTIVE ACTUAL BEST DESIGN",
        actual_best
    )

    top_results = []

    for k in TOP_K_VALUES:

        selected = evaluate_top_k(
            ranked,
            k
        )

        actual_best_cost = float(
            actual_best[TARGET]
        )

        selected_cost = float(
            selected[TARGET]
        )

        difference, percentage = (
            calculate_regret(
                actual_best_cost,
                selected_cost
            )
        )

        recovered = (
            abs(
                selected_cost
                - actual_best_cost
            )
            < 1e-9
        )

        reduction = (
            1.0
            - k / len(ranked)
        ) * 100.0

        result = {
            "top_k": k,

            "selected_actual_cost_INR":
                selected_cost,

            "actual_best_cost_INR":
                actual_best_cost,

            "difference_INR":
                difference,

            "difference_percent":
                percentage,

            "search_reduction_percent":
                reduction,

            "actual_best_recovered":
                bool(recovered),

            "ml_rank_of_selected_design":
                int(selected["ML_rank"]),

            "selected_design":
                selected.to_dict(),
        }

        top_results.append(
            result
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"TOP {k:,} ML CANDIDATES"
        )

        print(
            "=" * 70
        )

        print(
            f"Best actual cost found : "
            f"₹{selected_cost:,.2f}"
        )

        print(
            f"Exhaustive optimum     : "
            f"₹{actual_best_cost:,.2f}"
        )

        print(
            f"Difference             : "
            f"₹{difference:,.2f}"
        )

        print(
            f"Difference percentage  : "
            f"{percentage:.3f}%"
        )

        print(
            f"Search reduction       : "
            f"{reduction:.3f}%"
        )

        print(
            "Exhaustive optimum recovered: "
            f"{'YES' if recovered else 'NO'}"
        )

        print_design(
            f"BEST DESIGN WITHIN TOP {k:,}",
            selected
        )

    print()
    print(
        "=" * 70
    )
    print(
        "ML-ASSISTED OPTIMIZATION SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"Total feasible candidates : "
        f"{len(ranked):,}"
    )

    print(
        f"Exhaustive optimum         : "
        f"₹{actual_best[TARGET]:,.2f}"
    )

    print(
        f"Top-100 best actual cost   : "
        f"₹{top_results[0]['selected_actual_cost_INR']:,.2f}"
    )

    print(
        f"Top-500 best actual cost   : "
        f"₹{top_results[1]['selected_actual_cost_INR']:,.2f}"
    )

    print(
        f"Top-1000 best actual cost  : "
        f"₹{top_results[2]['selected_actual_cost_INR']:,.2f}"
    )

    print()

    print(
        "The ML model is used for "
        "candidate screening/ranking."
    )

    print(
        "Structural feasibility and actual "
        "cost remain the final decision criteria."
    )

    summary = {
        "total_feasible_candidates":
            int(len(ranked)),

        "actual_best_cost_INR":
            float(actual_best[TARGET]),

        "top_k_results": [
            {
                key: value
                for key, value in result.items()
                if key != "selected_design"
            }
            for result in top_results
        ],

        "method":
            "ML-assisted candidate screening followed "
            "by exact actual-cost selection",

        "model_file":
            str(MODEL_FILE),

        "candidate_file":
            str(CANDIDATE_FILE),

        "model_features":
            MODEL_FEATURES,

        "top_k_values":
            TOP_K_VALUES,

        "random_state":
            RANDOM_STATE,
    }

    save_outputs(
        ranked,
        actual_best,
        top_results,
        summary
    )

    print()
    print(
        "=" * 70
    )
    print(
        "OUTPUT FILES"
    )
    print(
        "=" * 70
    )

    output_files = [
        "all_candidate_predictions.csv",
        "ranked_candidates.csv",
        "top_50_ml_designs.csv",
        "top_500_ml_candidates.csv",
        "top_k_optimization_results.csv",
        "optimization_comparison.csv",
        "actual_best_design.csv",
        "ml_optimal_design.csv",
        "optimization_summary.json",
    ]

    for filename in output_files:
        print(
            OUTPUT_DIR / filename
        )

    print()

    print(
        "Figures :",
        FIGURES_DIR
    )

    print()

    print(
        "=" * 70
    )

    print(
        "ML-ASSISTED OPTIMIZATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()