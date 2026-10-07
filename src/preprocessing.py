from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit


RANDOM_STATE = 42
PLOT_SAMPLE_SIZE = 20000

PROJECT_ROOT = Path(__file__).resolve().parent

INPUT_FILE = PROJECT_ROOT / "beam_dataset_ML_ready.csv"

OUTPUT_DIR = PROJECT_ROOT / "processed"
FIGURES_DIR = OUTPUT_DIR / "figures"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VALIDATION_FILE = OUTPUT_DIR / "validation.csv"
TEST_FILE = OUTPUT_DIR / "test.csv"


ENGINEERING_FEATURES = [
    "ADL_kNm",
    "LL_kNm",
    "span_cc_mm",
    "fck_MPa",
    "fy_MPa",
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
    "leff_mm",
]


TARGET = "total_cost_INR"


PLOT_FEATURES = [
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
    "total_cost_INR",
]


def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n{INPUT_FILE}\n\n"
            "Place beam_dataset_ML_ready.csv in the same folder "
            "as preprocessing.py."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Dataset: {INPUT_FILE}")
    print(f"Rows loaded: {len(df):,}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


def validate_columns(df):

    required_columns = set(
        ENGINEERING_FEATURES
        + MODEL_FEATURES
        + [TARGET]
    )

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:

        print("\nMissing required columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "\nThe dataset does not contain all required columns."
        )

    print("\nAll required columns are present.")


def clean_data(df):

    print("\n" + "=" * 70)
    print("DATA VALIDATION AND CLEANING")
    print("=" * 70)

    original_rows = len(df)

    duplicate_count = df.duplicated().sum()

    print(f"Exact duplicate rows: {duplicate_count:,}")

    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    numeric_columns = list(
        dict.fromkeys(
            ENGINEERING_FEATURES
            + MODEL_FEATURES
            + [TARGET]
        )
    )

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    missing_mask = (
        df[numeric_columns]
        .isna()
        .any(axis=1)
    )

    missing_count = missing_mask.sum()

    print(
        f"Rows with missing required values: "
        f"{missing_count:,}"
    )

    if missing_count > 0:
        df = df.loc[~missing_mask].copy()

    invalid_mask = (
        (df[TARGET] <= 0)
        | (df["span_cc_mm"] <= 0)
        | (df["b_mm"] <= 0)
        | (df["D_final_mm"] <= 0)
        | (df["d_final_mm"] <= 0)
        | (df["fck_MPa"] <= 0)
        | (df["fy_MPa"] <= 0)
        | (df["Ast_prov_mm2"] <= 0)
        | (df["phi_main_mm"] <= 0)
        | (df["phi_stirrup_mm"] <= 0)
        | (df["cover_mm"] <= 0)
    )

    invalid_count = invalid_mask.sum()

    print(
        f"Physically invalid rows: "
        f"{invalid_count:,}"
    )

    if invalid_count > 0:
        df = df.loc[~invalid_mask].copy()

    df = df.reset_index(drop=True)

    print(f"\nOriginal rows : {original_rows:,}")
    print(f"Final rows    : {len(df):,}")
    print(
        f"Rows removed  : "
        f"{original_rows - len(df):,}"
    )

    return df


def create_engineering_groups(df):

    print("\n" + "=" * 70)
    print("CREATING ENGINEERING PROBLEM GROUPS")
    print("=" * 70)

    print("\nEngineering problem definition:")

    for column in ENGINEERING_FEATURES:
        print(f"  - {column}")

    group_values = (
        df[ENGINEERING_FEATURES]
        .astype(str)
        .agg("|".join, axis=1)
    )

    df = df.copy()

    df["_engineering_group"] = pd.factorize(
        group_values,
        sort=True
    )[0]

    number_of_groups = (
        df["_engineering_group"].nunique()
    )

    print(
        f"\nUnique engineering problems: "
        f"{number_of_groups:,}"
    )

    return df


def grouped_train_validation_test_split(df):

    print("\n" + "=" * 70)
    print("GROUPED TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    groups = df["_engineering_group"]

    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=RANDOM_STATE
    )

    train_validation_idx, test_idx = next(
        test_splitter.split(
            df,
            groups=groups
        )
    )

    train_validation = (
        df.iloc[train_validation_idx]
        .copy()
    )

    test = (
        df.iloc[test_idx]
        .copy()
    )

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.17647058823529413,
        random_state=RANDOM_STATE
    )

    train_idx, validation_idx = next(
        validation_splitter.split(
            train_validation,
            groups=train_validation[
                "_engineering_group"
            ]
        )
    )

    train = (
        train_validation.iloc[train_idx]
        .copy()
    )

    validation = (
        train_validation.iloc[validation_idx]
        .copy()
    )

    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(
        f"Train rows      : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    print(
        f"Test rows       : {len(test):,}"
    )

    print()

    print(
        f"Train groups      : "
        f"{train['_engineering_group'].nunique():,}"
    )

    print(
        f"Validation groups : "
        f"{validation['_engineering_group'].nunique():,}"
    )

    print(
        f"Test groups       : "
        f"{test['_engineering_group'].nunique():,}"
    )

    return train, validation, test


def check_group_overlap(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("ENGINEERING-PROBLEM OVERLAP CHECK")
    print("=" * 70)

    train_groups = set(
        train["_engineering_group"]
    )

    validation_groups = set(
        validation["_engineering_group"]
    )

    test_groups = set(
        test["_engineering_group"]
    )

    train_validation_overlap = (
        train_groups & validation_groups
    )

    train_test_overlap = (
        train_groups & test_groups
    )

    validation_test_overlap = (
        validation_groups & test_groups
    )

    all_three_overlap = (
        train_groups
        & validation_groups
        & test_groups
    )

    print(
        f"Train ∩ Validation : "
        f"{len(train_validation_overlap):,}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_test_overlap):,}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(validation_test_overlap):,}"
    )

    print(
        f"All three          : "
        f"{len(all_three_overlap):,}"
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
        or all_three_overlap
    ):

        raise RuntimeError(
            "\nERROR: Engineering-problem overlap "
            "was detected between the datasets."
        )

    print("\nSUCCESS:")
    print(
        "ZERO ENGINEERING-PROBLEM OVERLAP "
        "BETWEEN TRAIN / VALIDATION / TEST."
    )


def remove_group_column(
    train,
    validation,
    test
):

    train = train.drop(
        columns=["_engineering_group"]
    )

    validation = validation.drop(
        columns=["_engineering_group"]
    )

    test = test.drop(
        columns=["_engineering_group"]
    )

    return train, validation, test


def save_datasets(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATASETS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    validation.to_csv(
        VALIDATION_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print(
        f"Train      : {TRAIN_FILE}"
    )

    print(
        f"Validation : {VALIDATION_FILE}"
    )

    print(
        f"Test       : {TEST_FILE}"
    )


def create_visualization_sample(df):

    sample_size = min(
        PLOT_SAMPLE_SIZE,
        len(df)
    )

    if len(df) > sample_size:

        sample = df.sample(
            n=sample_size,
            random_state=RANDOM_STATE
        )

    else:

        sample = df.copy()

    return sample.reset_index(drop=True)


def save_figure(filename):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = FIGURES_DIR / filename

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {filename}")


def generate_correlation_matrix(df):

    print("\nGenerating correlation matrix...")

    correlation_columns = [
        "ADL_kNm",
        "LL_kNm",
        "span_cc_mm",
        "b_mm",
        "D_final_mm",
        "d_final_mm",
        "fck_MPa",
        "fy_MPa",
        "Ast_prov_mm2",
        "n_bars",
        "bar_spacing_mm",
        "pt_prov_percent",
        "leff_mm",
        "total_cost_INR"
    ]

    correlation = (
        df[correlation_columns]
        .corr()
    )

    fig, ax = plt.subplots(
        figsize=(14, 11)
    )

    image = ax.imshow(
        correlation,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(correlation_columns))
    )

    ax.set_yticks(
        range(len(correlation_columns))
    )

    ax.set_xticklabels(
        correlation_columns,
        rotation=45,
        ha="right",
        fontsize=9
    )

    ax.set_yticklabels(
        correlation_columns,
        fontsize=9
    )

    for i in range(len(correlation_columns)):

        for j in range(len(correlation_columns)):

            value = correlation.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    ax.set_title(
        "Correlation Matrix of Beam Design Variables",
        fontsize=15,
        pad=15
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Pearson correlation"
    )

    plt.tight_layout()

    save_figure(
        "correlation_matrix.png"
    )


def generate_target_cost_distribution(df):

    print("Generating target cost distribution...")

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        df[TARGET],
        bins=80
    )

    ax.set_xlabel(
        "Total Cost (INR)"
    )

    ax.set_ylabel(
        "Number of Designs"
    )

    ax.set_title(
        "Distribution of Total Beam Design Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "target_cost_distribution.png"
    )


def generate_span_distribution(df):

    print("Generating span distribution...")

    span_counts = (
        df["span_cc_mm"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        span_counts.index.astype(str),
        span_counts.values
    )

    ax.set_xlabel(
        "Span (mm)"
    )

    ax.set_ylabel(
        "Number of Designs"
    )

    ax.set_title(
        "Distribution of Beam Span"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "span_distribution.png"
    )


def generate_width_depth_distribution(df):

    sample = create_visualization_sample(df)

    print("Generating width-depth relationship...")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        sample["b_mm"],
        sample["D_final_mm"],
        s=8,
        alpha=0.25
    )

    ax.set_xlabel(
        "Beam Width, b (mm)"
    )

    ax.set_ylabel(
        "Overall Depth, D (mm)"
    )

    ax.set_title(
        "Beam Width vs Overall Depth"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "width_depth_distribution.png"
    )


def generate_span_vs_cost(df):

    sample = create_visualization_sample(df)

    print("Generating span vs cost plot...")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        sample["span_cc_mm"],
        sample[TARGET],
        s=8,
        alpha=0.25
    )

    ax.set_xlabel(
        "Span (mm)"
    )

    ax.set_ylabel(
        "Total Cost (INR)"
    )

    ax.set_title(
        "Beam Span vs Total Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "span_vs_cost.png"
    )


def generate_load_vs_cost(df):

    sample = create_visualization_sample(df)

    sample = sample.copy()

    sample["total_load_kNm"] = (
        sample["ADL_kNm"]
        + sample["LL_kNm"]
    )

    print("Generating load vs cost plot...")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        sample["total_load_kNm"],
        sample[TARGET],
        s=8,
        alpha=0.25
    )

    ax.set_xlabel(
        "ADL + LL (kN/m)"
    )

    ax.set_ylabel(
        "Total Cost (INR)"
    )

    ax.set_title(
        "Total Applied Load vs Beam Design Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "load_vs_cost.png"
    )


def generate_fck_vs_cost(df):

    sample = create_visualization_sample(df)

    print("Generating concrete grade vs cost plot...")

    grouped = (
        sample.groupby("fck_MPa")[TARGET]
        .median()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        grouped.index,
        grouped.values,
        marker="o"
    )

    ax.set_xlabel(
        "Concrete Grade, fck (MPa)"
    )

    ax.set_ylabel(
        "Median Total Cost (INR)"
    )

    ax.set_title(
        "Concrete Strength vs Median Beam Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "fck_vs_cost.png"
    )


def generate_fy_vs_cost(df):

    sample = create_visualization_sample(df)

    print("Generating steel grade vs cost plot...")

    grouped = (
        sample.groupby("fy_MPa")[TARGET]
        .median()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.plot(
        grouped.index,
        grouped.values,
        marker="o"
    )

    ax.set_xlabel(
        "Steel Yield Strength, fy (MPa)"
    )

    ax.set_ylabel(
        "Median Total Cost (INR)"
    )

    ax.set_title(
        "Steel Strength vs Median Beam Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "fy_vs_cost.png"
    )


def generate_reinforcement_vs_cost(df):

    sample = create_visualization_sample(df)

    print("Generating reinforcement vs cost plot...")

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        sample["Ast_prov_mm2"],
        sample[TARGET],
        s=8,
        alpha=0.25
    )

    ax.set_xlabel(
        "Provided Reinforcement, Ast (mm²)"
    )

    ax.set_ylabel(
        "Total Cost (INR)"
    )

    ax.set_title(
        "Provided Reinforcement vs Total Cost"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    save_figure(
        "reinforcement_vs_cost.png"
    )


def generate_feature_distributions(df):

    print("Generating feature distributions...")

    columns = [
        "ADL_kNm",
        "LL_kNm",
        "span_cc_mm",
        "b_mm",
        "D_final_mm",
        "fck_MPa",
        "fy_MPa",
        "Ast_prov_mm2",
        "pt_prov_percent",
        "leff_mm"
    ]

    fig, axes = plt.subplots(
        5,
        2,
        figsize=(14, 22)
    )

    axes = axes.flatten()

    for index, column in enumerate(columns):

        axes[index].hist(
            df[column],
            bins=40
        )

        axes[index].set_title(
            column
        )

        axes[index].set_xlabel(
            column
        )

        axes[index].set_ylabel(
            "Count"
        )

        axes[index].grid(
            alpha=0.25
        )

    fig.suptitle(
        "Distribution of Major Beam Design Variables",
        fontsize=17,
        y=0.995
    )

    plt.tight_layout()

    save_figure(
        "feature_distributions.png"
    )


def generate_train_validation_test_distribution(df):

    print(
        "Generating train-validation-test "
        "distribution plot..."
    )

    group_counts = {
        "Training": len(df["train_marker"]),
        "Validation": len(df["validation_marker"]),
        "Test": len(df["test_marker"])
    }

    labels = list(group_counts.keys())
    values = list(group_counts.values())

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        labels,
        values
    )

    ax.set_ylabel(
        "Number of Designs"
    )

    ax.set_title(
        "Dataset Split"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    for index, value in enumerate(values):

        ax.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    save_figure(
        "dataset_split.png"
    )


def generate_all_visualizations(df):

    print("\n" + "=" * 70)
    print("GENERATING DATA ANALYSIS FIGURES")
    print("=" * 70)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sample = create_visualization_sample(df)

    print(
        f"\nVisualization sample size: "
        f"{len(sample):,}"
    )

    generate_correlation_matrix(
        sample
    )

    generate_target_cost_distribution(
        sample
    )

    generate_span_distribution(
        df
    )

    generate_width_depth_distribution(
        sample
    )

    generate_span_vs_cost(
        sample
    )

    generate_load_vs_cost(
        sample
    )

    generate_fck_vs_cost(
        sample
    )

    generate_fy_vs_cost(
        sample
    )

    generate_reinforcement_vs_cost(
        sample
    )

    generate_feature_distributions(
        sample
    )

    print(
        "\nAll figures saved to:"
    )

    print(
        FIGURES_DIR
    )


def print_final_summary(
    train,
    validation,
    test
):

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Train rows      : {len(train):,}"
    )

    print(
        f"Validation rows : {len(validation):,}"
    )

    print(
        f"Test rows       : {len(test):,}"
    )

    print(
        f"Total rows      : "
        f"{len(train) + len(validation) + len(test):,}"
    )

    print("\nModel features:")

    for feature in MODEL_FEATURES:
        print(f"  - {feature}")

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        "\nProcessed datasets:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nAnalysis figures:"
    )

    print(
        FIGURES_DIR
    )

    print("=" * 70)


def main():

    df = load_data()

    validate_columns(
        df
    )

    df = clean_data(
        df
    )

    generate_all_visualizations(
        df
    )

    df = create_engineering_groups(
        df
    )

    train, validation, test = (
        grouped_train_validation_test_split(
            df
        )
    )

    check_group_overlap(
        train,
        validation,
        test
    )

    train, validation, test = (
        remove_group_column(
            train,
            validation,
            test
        )
    )

    save_datasets(
        train,
        validation,
        test
    )

    print_final_summary(
        train,
        validation,
        test
    )


if __name__ == "__main__":
    main()