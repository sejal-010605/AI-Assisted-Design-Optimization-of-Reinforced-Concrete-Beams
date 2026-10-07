from pathlib import Path
import importlib.util
import itertools
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

OUTPUT_DIR = PROJECT_ROOT / "processed"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD ENGINEERING CHECKER
# ============================================================

ENGINEERING_CHECKS_PATH = PROJECT_ROOT / "engineering_checks.py"

spec = importlib.util.spec_from_file_location(
    "engineering_checks",
    ENGINEERING_CHECKS_PATH
)

engineering_checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engineering_checks)

check_beam = engineering_checks.check_beam


# ============================================================
# 3. ENGINEERING PROBLEM
# ============================================================

ADL_kNm = 5.0
LL_kNm = 5.0
SPAN_MM = 5000.0
FCK_MPa = 25.0
FY_MPa = 500.0

EXPOSURE = "moderate"


# ============================================================
# 4. DESIGN PARAMETERS
# ============================================================

CONCRETE_DENSITY_KN_M3 = 25.0

FY_STIRRUP_MPa = 415.0
STIRRUP_LEGS = 2

NOMINAL_COVER_MM = 30.0

DEAD_LOAD_FACTOR = 1.5
LIVE_LOAD_FACTOR = 1.5

DEFLECTION_MODIFICATION_FACTOR = 1.0

SUPPORT_CONDITION = "simply_supported"


# ============================================================
# 5. COST PARAMETERS
# ============================================================

CONCRETE_COST_INR_M3 = 8000.0
STEEL_COST_INR_KG = 75.0

STEEL_DENSITY_KG_M3 = 7850.0


# ============================================================
# 6. DESIGN SPACE
# ============================================================

WIDTHS_MM = list(range(200, 401, 25))

DEPTHS_MM = list(range(300, 701, 25))

MAIN_BAR_DIAMETERS_MM = [
    12,
    16,
    20,
    25,
    28,
    32
]

NUMBER_OF_MAIN_BARS = list(range(2, 9))

STIRRUP_DIAMETERS_MM = [
    8,
    10,
    12
]

STIRRUP_SPACINGS_MM = list(range(100, 301, 25))


# ============================================================
# 7. THEORETICAL DESIGN COUNT
# ============================================================

THEORETICAL_COMBINATIONS = (
    len(WIDTHS_MM)
    * len(DEPTHS_MM)
    * len(MAIN_BAR_DIAMETERS_MM)
    * len(NUMBER_OF_MAIN_BARS)
    * len(STIRRUP_DIAMETERS_MM)
    * len(STIRRUP_SPACINGS_MM)
)


print("=" * 70)
print("RCC BEAM CANDIDATE GENERATION")
print("=" * 70)
print()

print("Engineering problem:")
print(f"ADL = {ADL_kNm} kN/m")
print(f"LL  = {LL_kNm} kN/m")
print(f"Span = {SPAN_MM:.0f} mm")
print(f"fck = {FCK_MPa:.0f} MPa")
print(f"fy  = {FY_MPa:.0f} MPa")
print(f"Exposure = {EXPOSURE}")
print()

print(
    f"Theoretical design combinations:\n"
    f"{THEORETICAL_COMBINATIONS:,}"
)
print()


# ============================================================
# 8. HELPER FUNCTIONS
# ============================================================

def bar_area_mm2(phi_mm):
    return math.pi * phi_mm ** 2 / 4.0


def steel_weight_kg(volume_m3):
    return volume_m3 * STEEL_DENSITY_KG_M3


# ============================================================
# 9. STORAGE AND COUNTERS
# ============================================================

candidate_rows = []

geometry_rejected = 0
feasible_count = 0


# ============================================================
# 10. MAIN CANDIDATE GENERATION LOOP
# ============================================================

for b_mm in WIDTHS_MM:

    width_evaluated = 0
    width_feasible = 0

    for (
        D_mm,
        phi_main_mm,
        n_bars,
        phi_stirrup_mm,
        stirrup_spacing_mm
    ) in itertools.product(
        DEPTHS_MM,
        MAIN_BAR_DIAMETERS_MM,
        NUMBER_OF_MAIN_BARS,
        STIRRUP_DIAMETERS_MM,
        STIRRUP_SPACINGS_MM
    ):

        # ----------------------------------------------------
        # Effective depth
        # ----------------------------------------------------

        d_mm = (
            D_mm
            - NOMINAL_COVER_MM
            - phi_stirrup_mm
            - phi_main_mm / 2.0
        )

        if d_mm <= 0:
            geometry_rejected += 1
            continue

        # ----------------------------------------------------
        # Main reinforcement area
        # ----------------------------------------------------

        Ast_mm2 = (
            n_bars
            * bar_area_mm2(phi_main_mm)
        )

        # ----------------------------------------------------
        # Reinforcement percentage
        # ----------------------------------------------------

        pt_percent = (
            100.0
            * Ast_mm2
            / (b_mm * d_mm)
        )

        # ----------------------------------------------------
        # Available width for main bars
        # ----------------------------------------------------

        internal_width_mm = (
            b_mm
            - 2.0 * NOMINAL_COVER_MM
            - 2.0 * phi_stirrup_mm
        )

        if internal_width_mm <= 0:
            geometry_rejected += 1
            continue

        # ----------------------------------------------------
        # Check horizontal bar fit
        # ----------------------------------------------------

        total_bar_diameter_mm = (
            n_bars * phi_main_mm
        )

        clear_width_available_mm = (
            internal_width_mm
            - total_bar_diameter_mm
        )

        if n_bars > 1:

            clear_spacing_mm = (
                clear_width_available_mm
                / (n_bars - 1)
            )

        else:

            clear_spacing_mm = (
                clear_width_available_mm
            )

        minimum_clear_spacing_mm = max(
            phi_main_mm,
            20.0
        )

        if clear_spacing_mm < minimum_clear_spacing_mm:
            geometry_rejected += 1
            continue

        # ----------------------------------------------------
        # Development length availability
        # ----------------------------------------------------

        available_development_length_mm = SPAN_MM

        # ----------------------------------------------------
        # Structural check
        # ----------------------------------------------------

        try:

            result = check_beam(
                ADL_kNm=ADL_kNm,
                LL_kNm=LL_kNm,
                span_mm=SPAN_MM,
                b_mm=float(b_mm),
                D_mm=float(D_mm),
                fck_MPa=FCK_MPa,
                fy_MPa=FY_MPa,
                Ast_prov_mm2=float(Ast_mm2),
                n_bars=int(n_bars),
                phi_main_mm=float(phi_main_mm),
                phi_stirrup_mm=float(phi_stirrup_mm),
                stirrup_legs=STIRRUP_LEGS,
                stirrup_spacing_mm=float(stirrup_spacing_mm),
                nominal_cover_mm=NOMINAL_COVER_MM,
                fy_stirrup_MPa=FY_STIRRUP_MPa,
                support_condition=SUPPORT_CONDITION,
                available_development_length_mm=(
                    available_development_length_mm
                ),
                dead_load_factor=DEAD_LOAD_FACTOR,
                live_load_factor=LIVE_LOAD_FACTOR,
                concrete_density_kN_m3=CONCRETE_DENSITY_KN_M3,
                deflection_modification_factor=(
                    DEFLECTION_MODIFICATION_FACTOR
                )
            )

        except Exception as exc:

            width_evaluated += 1

            row = {
                "ADL_kNm": ADL_kNm,
                "LL_kNm": LL_kNm,
                "span_cc_mm": SPAN_MM,
                "b_mm": b_mm,
                "D_final_mm": D_mm,
                "d_final_mm": d_mm,
                "fck_MPa": FCK_MPa,
                "fy_MPa": FY_MPa,
                "phi_main_mm": phi_main_mm,
                "phi_stirrup_mm": phi_stirrup_mm,
                "cover_mm": NOMINAL_COVER_MM,
                "Ast_prov_mm2": Ast_mm2,
                "n_bars": n_bars,
                "stirrup_spacing_mm": stirrup_spacing_mm,
                "pt_prov_percent": pt_percent,
                "leff_mm": SPAN_MM,
                "status": "INFEASIBLE",
                "engineering_status": "ERROR",
                "engineering_feasible": False,
                "feasible": False,
                "error": str(exc)
            }

            candidate_rows.append(row)

            continue

        width_evaluated += 1

        # ====================================================
        # FINAL ENGINEERING FEASIBILITY
        # ====================================================

        engineering_status = str(
            result.get(
                "status",
                ""
            )
        ).upper()

        if "feasible" in result:

            overall_pass = bool(
                result["feasible"]
            )

        elif "overall_pass" in result:

            overall_pass = bool(
                result["overall_pass"]
            )

        else:

            overall_pass = (
                engineering_status == "SAFE"
            )

        # ====================================================
        # COST CALCULATION
        # ====================================================

        # ----------------------------------------------------
        # Concrete
        # ----------------------------------------------------

        concrete_volume_m3 = (
            b_mm / 1000.0
            * D_mm / 1000.0
            * SPAN_MM / 1000.0
        )

        concrete_cost_INR = (
            concrete_volume_m3
            * CONCRETE_COST_INR_M3
        )

        # ----------------------------------------------------
        # Main reinforcement
        # ----------------------------------------------------

        main_steel_volume_m3 = (
            n_bars
            * bar_area_mm2(phi_main_mm)
            * SPAN_MM
            / 1e9
        )

        main_steel_weight_kg = steel_weight_kg(
            main_steel_volume_m3
        )

        # ----------------------------------------------------
        # Stirrup steel
        # ----------------------------------------------------

        stirrup_perimeter_mm = (
            2.0
            * (
                b_mm
                - NOMINAL_COVER_MM
                - phi_stirrup_mm / 2.0
            )
            +
            2.0
            * (
                D_mm
                - NOMINAL_COVER_MM
                - phi_stirrup_mm / 2.0
            )
        )

        number_of_stirrups = (
            math.floor(
                SPAN_MM
                / stirrup_spacing_mm
            )
            + 1
        )

        stirrup_total_length_m = (
            number_of_stirrups
            * stirrup_perimeter_mm
            / 1000.0
        )

        stirrup_steel_volume_m3 = (
            stirrup_total_length_m
            * bar_area_mm2(phi_stirrup_mm)
            / 1e6
        )

        stirrup_weight_kg = steel_weight_kg(
            stirrup_steel_volume_m3
        )

        # ----------------------------------------------------
        # Total steel and cost
        # ----------------------------------------------------

        total_steel_weight_kg = (
            main_steel_weight_kg
            + stirrup_weight_kg
        )

        steel_cost_INR = (
            total_steel_weight_kg
            * STEEL_COST_INR_KG
        )

        total_cost_INR = (
            concrete_cost_INR
            + steel_cost_INR
        )

        # ====================================================
        # FINAL STATUS
        # ====================================================

        if overall_pass:

            status = "FEASIBLE"

            feasible_count += 1
            width_feasible += 1

        else:

            status = "INFEASIBLE"

        # ====================================================
        # STORE CANDIDATE
        # ====================================================

        row = {
            "ADL_kNm": ADL_kNm,
            "LL_kNm": LL_kNm,
            "span_cc_mm": SPAN_MM,
            "b_mm": b_mm,
            "D_final_mm": D_mm,
            "d_final_mm": d_mm,
            "fck_MPa": FCK_MPa,
            "fy_MPa": FY_MPa,
            "phi_main_mm": phi_main_mm,
            "phi_stirrup_mm": phi_stirrup_mm,
            "cover_mm": NOMINAL_COVER_MM,
            "Ast_prov_mm2": Ast_mm2,
            "n_bars": n_bars,
            "stirrup_spacing_mm": stirrup_spacing_mm,
            "pt_prov_percent": pt_percent,
            "leff_mm": SPAN_MM,

            "status": status,

            "engineering_status":
                engineering_status,

            "engineering_feasible":
                overall_pass,

            "concrete_volume_m3":
                concrete_volume_m3,

            "concrete_cost_INR":
                concrete_cost_INR,

            "main_steel_weight_kg":
                main_steel_weight_kg,

            "stirrup_weight_kg":
                stirrup_weight_kg,

            "total_steel_weight_kg":
                total_steel_weight_kg,

            "steel_cost_INR":
                steel_cost_INR,

            "total_cost_INR":
                total_cost_INR,

            "feasible":
                overall_pass
        }

        # ====================================================
        # PRESERVE ENGINEERING CHECK OUTPUTS
        # ====================================================

        for key, value in result.items():

            if key in row:
                continue

            if np.isscalar(value):

                if isinstance(
                    value,
                    (np.bool_, bool)
                ):

                    row[key] = bool(value)

                elif isinstance(
                    value,
                    (np.integer, int)
                ):

                    row[key] = int(value)

                elif isinstance(
                    value,
                    (np.floating, float)
                ):

                    row[key] = float(value)

                elif isinstance(
                    value,
                    str
                ):

                    row[key] = value

        candidate_rows.append(row)

    print(
        f"Completed width b = {b_mm:3d} mm | "
        f"Evaluated = {width_evaluated:,} | "
        f"Feasible = {width_feasible:,}"
    )


# ============================================================
# 11. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(candidate_rows)


# ============================================================
# 12. SAVE ALL CANDIDATES
# ============================================================

all_candidates_path = (
    OUTPUT_DIR / "all_candidates.csv"
)

df.to_csv(
    all_candidates_path,
    index=False
)


# ============================================================
# 13. FILTER FEASIBLE CANDIDATES
# ============================================================

if "feasible" in df.columns:

    feasible_df = df[
        df["feasible"].astype(bool)
    ].copy()

else:

    feasible_df = df[
        df["status"] == "FEASIBLE"
    ].copy()


feasible_df = feasible_df.sort_values(
    by="total_cost_INR",
    ascending=True
).reset_index(drop=True)


feasible_path = (
    OUTPUT_DIR / "feasible_candidates.csv"
)

feasible_df.to_csv(
    feasible_path,
    index=False
)


# ============================================================
# 14. BEST DESIGN
# ============================================================

best_design_path = (
    OUTPUT_DIR / "best_design.csv"
)

if len(feasible_df) > 0:

    best_design = feasible_df.iloc[
        [0]
    ].copy()

    best_design.to_csv(
        best_design_path,
        index=False
    )

else:

    best_design = pd.DataFrame()

    best_design.to_csv(
        best_design_path,
        index=False
    )


# ============================================================
# 15. FINAL COUNTS
# ============================================================

evaluated_count = len(df)

geometry_rejected_count = geometry_rejected

structurally_evaluated = evaluated_count

structural_failures_count = (
    structurally_evaluated
    - feasible_count
)

if evaluated_count > 0:

    feasible_percentage = (
        feasible_count
        / evaluated_count
        * 100.0
    )

else:

    feasible_percentage = 0.0


# ============================================================
# 16. SUMMARY CSV
# ============================================================

summary = pd.DataFrame([
    {
        "theoretical_combinations":
            THEORETICAL_COMBINATIONS,

        "evaluated_candidates":
            evaluated_count,

        "geometry_rejected":
            geometry_rejected_count,

        "structurally_evaluated":
            structurally_evaluated,

        "structural_failures":
            structural_failures_count,

        "feasible_candidates":
            feasible_count,

        "feasible_percentage":
            feasible_percentage
    }
])


summary_path = (
    OUTPUT_DIR
    / "candidate_generation_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 17. PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("CANDIDATE GENERATION SUMMARY")
print("=" * 70)

print(
    f"Theoretical combinations : "
    f"{THEORETICAL_COMBINATIONS:,}"
)

print(
    f"Evaluated candidates     : "
    f"{evaluated_count:,}"
)

print(
    f"Geometry rejected        : "
    f"{geometry_rejected_count:,}"
)

print(
    f"Structural failures      : "
    f"{structural_failures_count:,}"
)

print(
    f"Feasible candidates      : "
    f"{feasible_count:,}"
)

print(
    f"Feasible % of evaluated  : "
    f"{feasible_percentage:.3f}%"
)

print()


# ============================================================
# 18. BEST DESIGN REPORT
# ============================================================

if len(best_design) > 0:

    best = best_design.iloc[0]

    print("=" * 70)
    print("OPTIMUM FEASIBLE DESIGN")
    print("=" * 70)

    print(
        f"Width b              : "
        f"{best['b_mm']:.0f} mm"
    )

    print(
        f"Overall depth D      : "
        f"{best['D_final_mm']:.0f} mm"
    )

    print(
        f"Effective depth d    : "
        f"{best['d_final_mm']:.2f} mm"
    )

    print(
        f"Main reinforcement   : "
        f"{int(best['n_bars'])} × "
        f"{best['phi_main_mm']:.0f} mm"
    )

    print(
        f"Stirrups             : "
        f"2-legged "
        f"{best['phi_stirrup_mm']:.0f} mm"
    )

    print(
        f"Stirrup spacing      : "
        f"{best['stirrup_spacing_mm']:.0f} mm"
    )

    print(
        f"Ast provided         : "
        f"{best['Ast_prov_mm2']:.2f} mm²"
    )

    print(
        f"Steel weight         : "
        f"{best['total_steel_weight_kg']:.2f} kg"
    )

    print(
        f"Concrete cost        : "
        f"₹{best['concrete_cost_INR']:,.2f}"
    )

    print(
        f"Steel cost           : "
        f"₹{best['steel_cost_INR']:,.2f}"
    )

    print(
        f"Total cost           : "
        f"₹{best['total_cost_INR']:,.2f}"
    )

    print()

else:

    print("=" * 70)
    print("NO FEASIBLE DESIGN FOUND.")
    print("=" * 70)
    print()


# ============================================================
# 19. FIGURE 1 — COST DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 6))

if len(df) > 0:

    plt.hist(
        df["total_cost_INR"].dropna(),
        bins=50
    )

plt.xlabel("Total Cost (INR)")
plt.ylabel("Number of Candidates")
plt.title("RCC Beam Candidate Cost Distribution")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "candidate_cost_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. FIGURE 2 — DEPTH VS COST
# ============================================================

plt.figure(figsize=(9, 6))

if len(feasible_df) > 0:

    plt.scatter(
        feasible_df["D_final_mm"],
        feasible_df["total_cost_INR"],
        s=8,
        alpha=0.5
    )

plt.xlabel("Overall Depth D (mm)")
plt.ylabel("Total Cost (INR)")
plt.title(
    "Beam Depth vs Total Cost — Feasible Designs"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "depth_vs_cost.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. FIGURE 3 — WIDTH VS COST
# ============================================================

plt.figure(figsize=(9, 6))

if len(feasible_df) > 0:

    plt.scatter(
        feasible_df["b_mm"],
        feasible_df["total_cost_INR"],
        s=8,
        alpha=0.5
    )

plt.xlabel("Beam Width b (mm)")
plt.ylabel("Total Cost (INR)")
plt.title(
    "Beam Width vs Total Cost — Feasible Designs"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "width_vs_cost.png",
    dpi=300
)

plt.close()


# ============================================================
# 22. FIGURE 4 — TOP FEASIBLE DESIGNS
# ============================================================

plt.figure(figsize=(11, 7))

if len(feasible_df) > 0:

    top_n = min(
        20,
        len(feasible_df)
    )

    top = feasible_df.head(
        top_n
    ).copy()

    labels = []

    for _, row in top.iterrows():

        labels.append(
            f"{int(row['b_mm'])}×"
            f"{int(row['D_final_mm'])}, "
            f"{int(row['n_bars'])}Ø"
            f"{int(row['phi_main_mm'])}"
        )

    y = np.arange(top_n)

    plt.barh(
        y,
        top["total_cost_INR"]
    )

    plt.yticks(
        y,
        labels
    )

    plt.gca().invert_yaxis()

plt.xlabel("Total Cost (INR)")
plt.ylabel(
    "Beam Design (b × D, main reinforcement)"
)
plt.title(
    "Top 20 Cheapest Feasible Beam Designs"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "top_feasible_designs.png",
    dpi=300
)

plt.close()


# ============================================================
# 23. FINAL OUTPUT REPORT
# ============================================================

print("=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(
    f"All candidates       : "
    f"{all_candidates_path}"
)

print(
    f"Feasible candidates  : "
    f"{feasible_path}"
)

print(
    f"Best design          : "
    f"{best_design_path}"
)

print(
    f"Summary              : "
    f"{summary_path}"
)

print(
    f"Figures              : "
    f"{FIGURE_DIR}"
)

print()
print(
    "Candidate generation completed successfully."
)