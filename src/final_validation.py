from pathlib import Path
import json
import importlib.util
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "processed"

CANDIDATE_FILE = (
    PROCESSED_DIR / "feasible_candidates.csv"
)

OPTIMIZATION_FILE = (
    PROCESSED_DIR
    / "ml_optimization_outputs"
    / "ml_optimal_design.csv"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "final_validation_outputs"
)

RESULT_FILE = (
    OUTPUT_DIR / "final_validation_results.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "final_validation_summary.json"
)


def load_engineering_checks():

    file_path = PROJECT_ROOT / "engineering_checks.py"

    if not file_path.exists():
        raise FileNotFoundError(
            f"engineering_checks.py not found:\n{file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "engineering_checks",
        file_path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


def load_designs():

    if not CANDIDATE_FILE.exists():
        raise FileNotFoundError(
            f"Candidate file not found:\n{CANDIDATE_FILE}"
        )

    if not OPTIMIZATION_FILE.exists():
        raise FileNotFoundError(
            f"ML optimal design file not found:\n{OPTIMIZATION_FILE}"
        )

    candidates = pd.read_csv(
        CANDIDATE_FILE
    )

    ml_designs = pd.read_csv(
        OPTIMIZATION_FILE
    )

    if ml_designs.empty:
        raise ValueError(
            "ml_optimal_design.csv is empty."
        )

    actual_best_idx = candidates[
        "total_cost_INR"
    ].idxmin()

    actual_best = candidates.loc[
        actual_best_idx
    ]

    ml_design = ml_designs.iloc[0]

    return actual_best, ml_design


def run_engineering_check(checks, row):

    return checks.check_beam(

        ADL_kNm=float(
            row["ADL_kNm"]
        ),

        LL_kNm=float(
            row["LL_kNm"]
        ),

        span_mm=float(
            row["span_cc_mm"]
        ),

        b_mm=float(
            row["b_mm"]
        ),

        D_mm=float(
            row["D_final_mm"]
        ),

        fck_MPa=float(
            row["fck_MPa"]
        ),

        fy_MPa=float(
            row["fy_MPa"]
        ),

        Ast_prov_mm2=float(
            row["Ast_prov_mm2"]
        ),

        n_bars=int(
            row["n_bars"]
        ),

        phi_main_mm=float(
            row["phi_main_mm"]
        ),

        phi_stirrup_mm=float(
            row["phi_stirrup_mm"]
        ),

        stirrup_legs=2,

        stirrup_spacing_mm=float(
            row["stirrup_spacing_mm"]
        ),

        nominal_cover_mm=float(
            row["cover_mm"]
        ),

        fy_stirrup_MPa=415.0,

        support_condition="simply_supported",

        available_development_length_mm=None,

        dead_load_factor=1.5,

        live_load_factor=1.5,

        concrete_density_kN_m3=25.0,

        deflection_modification_factor=1.0
    )


def design_summary(name, row, result):

    return {
        "design": name,

        "b_mm": float(
            row["b_mm"]
        ),

        "D_mm": float(
            row["D_final_mm"]
        ),

        "d_mm": float(
            row["d_final_mm"]
        ),

        "phi_main_mm": float(
            row["phi_main_mm"]
        ),

        "n_bars": int(
            row["n_bars"]
        ),

        "phi_stirrup_mm": float(
            row["phi_stirrup_mm"]
        ),

        "stirrup_spacing_mm": float(
            row["stirrup_spacing_mm"]
        ),

        "Ast_prov_mm2": float(
            row["Ast_prov_mm2"]
        ),

        "actual_cost_INR": float(
            row["total_cost_INR"]
        ),

        "feasible": bool(
            result.get(
                "feasible",
                False
            )
        ),

        "status": str(
            result.get(
                "status",
                ""
            )
        ),

        "failure_reasons": result.get(
            "failure_reasons",
            []
        ),

        "full_result": result
    }


def print_design_info(name, row):

    print()
    print("=" * 70)
    print(
        f"{name}"
    )
    print("=" * 70)

    print(
        f"Width b                 : "
        f"{float(row['b_mm']):.0f} mm"
    )

    print(
        f"Overall depth D         : "
        f"{float(row['D_final_mm']):.0f} mm"
    )

    print(
        f"Effective depth d       : "
        f"{float(row['d_final_mm']):.2f} mm"
    )

    print(
        f"Main reinforcement      : "
        f"{int(row['n_bars'])} × "
        f"{float(row['phi_main_mm']):.0f} mm"
    )

    print(
        f"Stirrups                : "
        f"2-legged "
        f"{float(row['phi_stirrup_mm']):.0f} mm "
        f"@ "
        f"{float(row['stirrup_spacing_mm']):.0f} mm"
    )

    print(
        f"Ast provided            : "
        f"{float(row['Ast_prov_mm2']):.2f} mm²"
    )

    print(
        f"Actual cost             : "
        f"₹{float(row['total_cost_INR']):,.2f}"
    )


def main():

    print("=" * 70)
    print("FINAL RCC BEAM VALIDATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    checks = load_engineering_checks()

    actual_best, ml_design = load_designs()

    print()
    print(
        "Running independent engineering validation..."
    )

    actual_result = run_engineering_check(
        checks,
        actual_best
    )

    ml_result = run_engineering_check(
        checks,
        ml_design
    )

    print_design_info(
        "EXHAUSTIVE OPTIMUM",
        actual_best
    )

    print()

    print(
        "AUTHORITATIVE ENGINEERING CHECK REPORT"
    )

    print("-" * 70)

    checks.print_check_report(
        actual_result
    )

    print_design_info(
        "ML-ASSISTED OPTIMUM",
        ml_design
    )

    print()

    print(
        "AUTHORITATIVE ENGINEERING CHECK REPORT"
    )

    print("-" * 70)

    checks.print_check_report(
        ml_result
    )

    actual_cost = float(
        actual_best["total_cost_INR"]
    )

    ml_cost = float(
        ml_design["total_cost_INR"]
    )

    difference = (
        ml_cost - actual_cost
    )

    difference_percent = (
        difference
        / actual_cost
        * 100.0
    )

    same_design = all(
        abs(
            float(actual_best[col])
            - float(ml_design[col])
        ) < 1e-9
        for col in [
            "b_mm",
            "D_final_mm",
            "phi_main_mm",
            "n_bars",
            "phi_stirrup_mm",
            "stirrup_spacing_mm"
        ]
    )

    actual_feasible = bool(
        actual_result.get(
            "feasible",
            False
        )
    )

    ml_feasible = bool(
        ml_result.get(
            "feasible",
            False
        )
    )

    actual_summary = design_summary(
        "Exhaustive optimum",
        actual_best,
        actual_result
    )

    ml_summary = design_summary(
        "ML-assisted optimum",
        ml_design,
        ml_result
    )

    comparison = {

        "exhaustive_optimum_cost_INR":
            actual_cost,

        "ml_assisted_optimum_cost_INR":
            ml_cost,

        "cost_difference_INR":
            difference,

        "cost_difference_percent":
            difference_percent,

        "same_design":
            same_design,

        "exhaustive_optimum_feasible":
            actual_feasible,

        "ml_assisted_optimum_feasible":
            ml_feasible,

        "both_designs_feasible":
            actual_feasible and ml_feasible
    }

    results_table = pd.DataFrame([
        {
            "design":
                "Exhaustive optimum",

            "b_mm":
                actual_summary["b_mm"],

            "D_mm":
                actual_summary["D_mm"],

            "d_mm":
                actual_summary["d_mm"],

            "phi_main_mm":
                actual_summary["phi_main_mm"],

            "n_bars":
                actual_summary["n_bars"],

            "phi_stirrup_mm":
                actual_summary[
                    "phi_stirrup_mm"
                ],

            "stirrup_spacing_mm":
                actual_summary[
                    "stirrup_spacing_mm"
                ],

            "Ast_prov_mm2":
                actual_summary[
                    "Ast_prov_mm2"
                ],

            "actual_cost_INR":
                actual_cost,

            "feasible":
                actual_feasible,

            "status":
                actual_summary["status"],

            "failure_reasons":
                "; ".join(
                    map(
                        str,
                        actual_summary[
                            "failure_reasons"
                        ]
                    )
                )
        },

        {
            "design":
                "ML-assisted optimum",

            "b_mm":
                ml_summary["b_mm"],

            "D_mm":
                ml_summary["D_mm"],

            "d_mm":
                ml_summary["d_mm"],

            "phi_main_mm":
                ml_summary["phi_main_mm"],

            "n_bars":
                ml_summary["n_bars"],

            "phi_stirrup_mm":
                ml_summary[
                    "phi_stirrup_mm"
                ],

            "stirrup_spacing_mm":
                ml_summary[
                    "stirrup_spacing_mm"
                ],

            "Ast_prov_mm2":
                ml_summary[
                    "Ast_prov_mm2"
                ],

            "actual_cost_INR":
                ml_cost,

            "feasible":
                ml_feasible,

            "status":
                ml_summary["status"],

            "failure_reasons":
                "; ".join(
                    map(
                        str,
                        ml_summary[
                            "failure_reasons"
                        ]
                    )
                )
        }
    ])

    results_table.to_csv(
        RESULT_FILE,
        index=False
    )

    summary = {

        "purpose":
            "Independent validation of the "
            "exhaustive and ML-assisted optimum "
            "using engineering_checks.py.",

        "exhaustive_optimum":
            actual_summary,

        "ml_assisted_optimum":
            ml_summary,

        "comparison":
            comparison
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            default=str
        )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(
        f"Exhaustive optimum cost : "
        f"₹{actual_cost:,.2f}"
    )

    print(
        f"ML-assisted cost        : "
        f"₹{ml_cost:,.2f}"
    )

    print(
        f"Difference              : "
        f"₹{difference:,.2f}"
    )

    print(
        f"Difference percentage   : "
        f"{difference_percent:.3f}%"
    )

    print(
        f"Same design             : "
        f"{'YES' if same_design else 'NO'}"
    )

    print()

    print(
        "AUTHORITATIVE FEASIBILITY"
    )

    print(
        f"Exhaustive optimum      : "
        f"{'PASS' if actual_feasible else 'FAIL'}"
    )

    print(
        f"ML-assisted optimum     : "
        f"{'PASS' if ml_feasible else 'FAIL'}"
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        RESULT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()