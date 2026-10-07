from math import pi
from typing import Dict, Any


# ============================================================
# RCC BEAM ENGINEERING CHECKS
# IS 456:2000 - RECTANGULAR SINGLY REINFORCED BEAM
# ============================================================
#
# Units:
#   Length  : mm
#   Force   : N
#   Stress  : N/mm2
#   Moment  : N-mm
#   Load    : kN/m
#
# Scope:
#   - Simply supported rectangular beam
#   - Singly reinforced flexure
#   - Shear
#   - Minimum shear reinforcement
#   - Stirrup spacing
#   - Deflection span/depth screening
#   - Development length
#   - Longitudinal reinforcement detailing
#
# Important:
# This is an engineering screening/design calculation engine
# for the optimization project. Final structural design must
# still be independently checked by a qualified structural
# engineer against the project drawings, load combinations,
# support conditions, seismic requirements, exposure,
# anchorage, and applicable amendments.
# ============================================================


# ============================================================
# DEFAULT PARAMETERS
# ============================================================

DEFAULT_PARAMS = {
    "concrete_density_kN_m3": 25.0,

    "nominal_cover_mm": 30.0,

    "fy_stirrup_MPa": 415.0,

    "stirrup_legs": 2,

    "stirrup_spacing_mm": 200.0,

    "support_condition": "simply_supported",

    "dead_load_factor": 1.5,

    "live_load_factor": 1.5,

    "deflection_modification_factor": 1.0,

    "minimum_stirrup_diameter_mm": 8.0,

    "maximum_stirrup_spacing_mm": 300.0,

    "minimum_clear_bar_spacing_mm": 25.0,

    "maximum_tension_steel_ratio": 0.04,

    "minimum_main_bar_diameter_mm": 12.0,
}


# ============================================================
# BASIC VALIDATION
# ============================================================

def validate_positive(
    value: float,
    name: str
) -> None:

    if value <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )


def validate_materials(
    fck_MPa: float,
    fy_MPa: float
) -> None:

    validate_positive(
        fck_MPa,
        "fck_MPa"
    )

    validate_positive(
        fy_MPa,
        "fy_MPa"
    )


# ============================================================
# MATERIAL PROPERTIES
# ============================================================

def get_xu_max_ratio(
    fy_MPa: float
) -> float:
    """
    IS 456 limiting neutral axis depth ratio xu,max / d.

    Fe250 -> 0.53
    Fe415 -> 0.48
    Fe500 -> 0.46

    Linear interpolation is used between these grades.
    """

    if fy_MPa <= 250.0:
        return 0.53

    if fy_MPa <= 415.0:

        return (
            0.53
            + (fy_MPa - 250.0)
            / (415.0 - 250.0)
            * (0.48 - 0.53)
        )

    if fy_MPa <= 500.0:

        return (
            0.48
            + (fy_MPa - 415.0)
            / (500.0 - 415.0)
            * (0.46 - 0.48)
        )

    return 0.46


def get_tau_c_max(
    fck_MPa: float
) -> float:
    """
    IS 456 Table 20.
    Maximum nominal shear stress for beams.
    """

    table = {
        15.0: 2.5,
        20.0: 2.8,
        25.0: 3.1,
        30.0: 3.5,
        35.0: 3.7,
        40.0: 4.0,
        45.0: 4.0,
        50.0: 4.0,
    }

    grades = sorted(table)

    if fck_MPa <= grades[0]:
        return table[grades[0]]

    if fck_MPa >= grades[-1]:
        return table[grades[-1]]

    for lower, upper in zip(
        grades[:-1],
        grades[1:]
    ):

        if lower <= fck_MPa <= upper:

            return (
                table[lower]
                + (
                    (fck_MPa - lower)
                    / (upper - lower)
                )
                * (
                    table[upper]
                    - table[lower]
                )
            )

    return table[grades[-1]]


def get_tau_c_table_value(
    fck_MPa: float,
    pt_percent: float
) -> float:
    """
    IS 456 Table 19 interpolation.

    pt_percent:
        percentage of longitudinal tension steel
        based on 100 Ast / (b d).

    Table is interpolated in:
        1. reinforcement percentage
        2. concrete grade
    """

    pt_levels = [
        0.15,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        1.75,
        2.00,
        2.50,
        3.00,
    ]

    table = {

        15.0: [
            0.28,
            0.35,
            0.46,
            0.54,
            0.60,
            0.64,
            0.68,
            0.71,
            0.71,
            0.71,
            0.71,
        ],

        20.0: [
            0.28,
            0.36,
            0.48,
            0.56,
            0.62,
            0.67,
            0.71,
            0.75,
            0.79,
            0.82,
            0.85,
        ],

        25.0: [
            0.29,
            0.39,
            0.51,
            0.60,
            0.66,
            0.71,
            0.76,
            0.80,
            0.84,
            0.88,
            0.92,
        ],

        30.0: [
            0.30,
            0.40,
            0.52,
            0.62,
            0.69,
            0.74,
            0.79,
            0.83,
            0.87,
            0.92,
            0.96,
        ],

        35.0: [
            0.31,
            0.41,
            0.54,
            0.64,
            0.71,
            0.76,
            0.81,
            0.85,
            0.89,
            0.94,
            0.98,
        ],

        40.0: [
            0.32,
            0.42,
            0.56,
            0.66,
            0.74,
            0.79,
            0.83,
            0.88,
            0.92,
            0.97,
            1.01,
        ],
    }

    grades = sorted(table)

    pt = max(
        pt_levels[0],
        min(
            float(pt_percent),
            pt_levels[-1]
        )
    )

    def interpolate_pt(
        values
    ) -> float:

        if pt <= pt_levels[0]:
            return values[0]

        if pt >= pt_levels[-1]:
            return values[-1]

        for i in range(
            len(pt_levels) - 1
        ):

            p1 = pt_levels[i]
            p2 = pt_levels[i + 1]

            if p1 <= pt <= p2:

                v1 = values[i]
                v2 = values[i + 1]

                return (
                    v1
                    + (
                        (pt - p1)
                        / (p2 - p1)
                    )
                    * (v2 - v1)
                )

        return values[-1]

    values_by_grade = [
        interpolate_pt(table[grade])
        for grade in grades
    ]

    if fck_MPa <= grades[0]:
        return values_by_grade[0]

    if fck_MPa >= grades[-1]:
        return values_by_grade[-1]

    for i in range(
        len(grades) - 1
    ):

        f1 = grades[i]
        f2 = grades[i + 1]

        if f1 <= fck_MPa <= f2:

            v1 = values_by_grade[i]
            v2 = values_by_grade[i + 1]

            return (
                v1
                + (
                    (fck_MPa - f1)
                    / (f2 - f1)
                )
                * (v2 - v1)
            )

    return values_by_grade[-1]


def get_tau_bd(
    fck_MPa: float,
    deformed_bar: bool = True
) -> float:
    """
    IS 456 Table 26.2.1.1 design bond stress.

    Base values are for plain bars in tension.
    Deformed bars receive 60% enhancement.
    """

    table = {
        15.0: 1.0,
        20.0: 1.2,
        25.0: 1.4,
        30.0: 1.5,
        35.0: 1.7,
        40.0: 1.9,
    }

    grades = sorted(table)

    if fck_MPa <= grades[0]:

        value = table[grades[0]]

    elif fck_MPa >= grades[-1]:

        value = table[grades[-1]]

    else:

        value = table[grades[-1]]

        for lower, upper in zip(
            grades[:-1],
            grades[1:]
        ):

            if lower <= fck_MPa <= upper:

                value = (
                    table[lower]
                    + (
                        (fck_MPa - lower)
                        / (upper - lower)
                    )
                    * (
                        table[upper]
                        - table[lower]
                    )
                )

                break

    if deformed_bar:
        value *= 1.60

    return value


# ============================================================
# GEOMETRY
# ============================================================

def calculate_effective_depth(
    D_mm: float,
    nominal_cover_mm: float,
    phi_main_mm: float
) -> float:

    return (
        D_mm
        - nominal_cover_mm
        - phi_main_mm / 2.0
    )


def calculate_longitudinal_bar_spacing(
    b_mm: float,
    nominal_cover_mm: float,
    phi_stirrup_mm: float,
    phi_main_mm: float,
    n_bars: int
) -> Dict[str, float]:

    if n_bars < 2:

        return {
            "bar_center_spacing_mm": 0.0,
            "bar_clear_spacing_mm": 0.0,
        }

    available_centerline_width = (
        b_mm
        - 2.0 * nominal_cover_mm
        - 2.0 * phi_stirrup_mm
        - phi_main_mm
    )

    center_spacing = (
        available_centerline_width
        / (n_bars - 1)
    )

    clear_spacing = (
        center_spacing
        - phi_main_mm
    )

    return {

        "bar_center_spacing_mm": (
            center_spacing
        ),

        "bar_clear_spacing_mm": (
            clear_spacing
        ),
    }


# ============================================================
# LOADS
# ============================================================

def calculate_self_weight(
    b_mm: float,
    D_mm: float,
    concrete_density_kN_m3: float = 25.0
) -> float:

    area_m2 = (
        b_mm
        * D_mm
        / 1_000_000.0
    )

    return (
        area_m2
        * concrete_density_kN_m3
    )


def calculate_design_loads(
    ADL_kNm: float,
    LL_kNm: float,
    self_weight_kNm: float,
    dead_load_factor: float = 1.5,
    live_load_factor: float = 1.5
) -> Dict[str, float]:

    total_dead_load = (
        ADL_kNm
        + self_weight_kNm
    )

    factored_load = (
        dead_load_factor
        * total_dead_load
        +
        live_load_factor
        * LL_kNm
    )

    return {

        "self_weight_kNm": (
            self_weight_kNm
        ),

        "total_dead_load_kNm": (
            total_dead_load
        ),

        "factored_load_kNm": (
            factored_load
        ),
    }


def calculate_simple_span_actions(
    factored_load_kNm: float,
    span_mm: float
) -> Dict[str, float]:

    span_m = (
        span_mm / 1000.0
    )

    Mu_kNm = (
        factored_load_kNm
        * span_m ** 2
        / 8.0
    )

    Vu_kN = (
        factored_load_kNm
        * span_m
        / 2.0
    )

    return {

        "Mu_kNm": Mu_kNm,

        "Vu_kN": Vu_kN,
    }


# ============================================================
# FLEXURE
# ============================================================

def calculate_limiting_flexural_capacity(
    b_mm: float,
    d_mm: float,
    fck_MPa: float,
    fy_MPa: float
) -> Dict[str, float]:

    xu_max_ratio = (
        get_xu_max_ratio(fy_MPa)
    )

    xu_max = (
        xu_max_ratio
        * d_mm
    )

    Mu_lim_Nmm = (
        0.36
        * fck_MPa
        * b_mm
        * xu_max
        * (
            d_mm
            - 0.42 * xu_max
        )
    )

    Mu_lim_kNm = (
        Mu_lim_Nmm
        / 1_000_000.0
    )

    Ast_lim = (
        0.36
        * fck_MPa
        * b_mm
        * xu_max
        / (
            0.87
            * fy_MPa
        )
    )

    return {

        "xu_max_ratio": (
            xu_max_ratio
        ),

        "xu_max_mm": xu_max,

        "Mu_lim_kNm": (
            Mu_lim_kNm
        ),

        "Ast_lim_mm2": (
            Ast_lim
        ),
    }


def check_flexure(
    b_mm: float,
    D_mm: float,
    d_mm: float,
    Ast_prov_mm2: float,
    fck_MPa: float,
    fy_MPa: float,
    Mu_kNm: float
) -> Dict[str, Any]:

    validate_positive(
        Ast_prov_mm2,
        "Ast_prov_mm2"
    )

    # --------------------------------------------------------
    # Neutral axis from supplied tension steel
    # --------------------------------------------------------

    xu = (
        0.87
        * fy_MPa
        * Ast_prov_mm2
        / (
            0.36
            * fck_MPa
            * b_mm
        )
    )

    # --------------------------------------------------------
    # Limiting section
    # --------------------------------------------------------

    limiting = (
        calculate_limiting_flexural_capacity(
            b_mm,
            d_mm,
            fck_MPa,
            fy_MPa
        )
    )

    xu_max = (
        limiting["xu_max_mm"]
    )

    Mu_lim_kNm = (
        limiting["Mu_lim_kNm"]
    )

    Ast_lim_mm2 = (
        limiting["Ast_lim_mm2"]
    )

    # --------------------------------------------------------
    # Nominal moment from supplied steel
    #
    # This value is reported diagnostically.
    # It is NOT treated as an acceptable design capacity
    # when xu > xu,max.
    # --------------------------------------------------------

    Mu_nominal_Nmm = (
        0.87
        * fy_MPa
        * Ast_prov_mm2
        * (
            d_mm
            - 0.42 * xu
        )
    )

    Mu_nominal_kNm = (
        Mu_nominal_Nmm
        / 1_000_000.0
    )

    # --------------------------------------------------------
    # Reinforcement limits
    # --------------------------------------------------------

    Ast_min_mm2 = (
        0.85
        * b_mm
        * d_mm
        / fy_MPa
    )

    Ast_max_mm2 = (
        0.04
        * b_mm
        * D_mm
    )

    pt_percent = (
        100.0
        * Ast_prov_mm2
        / (
            b_mm
            * d_mm
        )
    )

    # --------------------------------------------------------
    # Individual checks
    # --------------------------------------------------------

    neutral_axis_pass = (
        xu <= xu_max
    )

    minimum_steel_pass = (
        Ast_prov_mm2
        >= Ast_min_mm2
    )

    maximum_steel_pass = (
        Ast_prov_mm2
        <= Ast_max_mm2
    )

    limiting_capacity_pass = (
        Mu_kNm
        <= Mu_lim_kNm
    )

    # A singly reinforced section is not accepted if it is
    # beyond xu,max, even if the raw nominal moment expression
    # happens to exceed Mu.
    overall_pass = (
        neutral_axis_pass
        and minimum_steel_pass
        and maximum_steel_pass
        and limiting_capacity_pass
    )

    # --------------------------------------------------------
    # Failure reason
    # --------------------------------------------------------

    reasons = []

    if not neutral_axis_pass:

        reasons.append(
            "xu exceeds xu,max "
            "(over-reinforced section)"
        )

    if not minimum_steel_pass:

        reasons.append(
            "tension reinforcement below minimum"
        )

    if not maximum_steel_pass:

        reasons.append(
            "tension reinforcement exceeds 4% of bD"
        )

    if not limiting_capacity_pass:

        reasons.append(
            "factored moment exceeds limiting "
            "singly-reinforced capacity"
        )

    if not reasons:

        reason = "OK"

    else:

        reason = "; ".join(
            reasons
        )

    utilization = (
        Mu_kNm
        / Mu_lim_kNm
        if Mu_lim_kNm > 0
        else float("inf")
    )

    return {

        "pass": overall_pass,

        "reason": reason,

        "Mu_demand_kNm": Mu_kNm,

        "Mu_lim_kNm": Mu_lim_kNm,

        "Mu_nominal_diagnostic_kNm": (
            Mu_nominal_kNm
        ),

        "flexural_utilization": (
            utilization
        ),

        "xu_mm": xu,

        "xu_max_mm": xu_max,

        "xu_max_ratio": (
            limiting["xu_max_ratio"]
        ),

        "Ast_prov_mm2": (
            Ast_prov_mm2
        ),

        "Ast_min_mm2": (
            Ast_min_mm2
        ),

        "Ast_lim_mm2": (
            Ast_lim_mm2
        ),

        "Ast_max_mm2": (
            Ast_max_mm2
        ),

        "pt_percent": pt_percent,

        "neutral_axis_pass": (
            neutral_axis_pass
        ),

        "limiting_capacity_pass": (
            limiting_capacity_pass
        ),

        "minimum_steel_pass": (
            minimum_steel_pass
        ),

        "maximum_steel_pass": (
            maximum_steel_pass
        ),
    }


# ============================================================
# SHEAR
# ============================================================

def check_shear(
    b_mm: float,
    d_mm: float,
    Ast_prov_mm2: float,
    fck_MPa: float,
    Vu_kN: float,
    phi_stirrup_mm: float,
    stirrup_legs: int,
    fy_stirrup_MPa: float,
    stirrup_spacing_mm: float
) -> Dict[str, Any]:

    pt_percent = (
        100.0
        * Ast_prov_mm2
        / (
            b_mm
            * d_mm
        )
    )

    tau_v = (
        Vu_kN
        * 1000.0
        / (
            b_mm
            * d_mm
        )
    )

    tau_c = (
        get_tau_c_table_value(
            fck_MPa,
            pt_percent
        )
    )

    tau_c_max = (
        get_tau_c_max(
            fck_MPa
        )
    )

    maximum_shear_pass = (
        tau_v
        <= tau_c_max
    )

    concrete_shear_pass = (
        tau_v
        <= tau_c
    )

    # --------------------------------------------------------
    # Stirrup area
    # --------------------------------------------------------

    Asv = (
        stirrup_legs
        * pi
        * phi_stirrup_mm ** 2
        / 4.0
    )

    fyv = min(
        fy_stirrup_MPa,
        415.0
    )

    # --------------------------------------------------------
    # Shear carried by reinforcement
    # --------------------------------------------------------

    Vus_required_N = max(
        0.0,
        (
            Vu_kN * 1000.0
            -
            tau_c
            * b_mm
            * d_mm
        )
    )

    if Vus_required_N > 0:

        spacing_required_from_shear_mm = (
            0.87
            * fyv
            * Asv
            * d_mm
            / Vus_required_N
        )

        shear_reinforcement_required = True

    else:

        spacing_required_from_shear_mm = (
            float("inf")
        )

        shear_reinforcement_required = False

    # --------------------------------------------------------
    # Capacity of provided stirrups
    # --------------------------------------------------------

    Vus_provided_N = (
        0.87
        * fyv
        * Asv
        * d_mm
        / stirrup_spacing_mm
    )

    shear_reinforcement_capacity_pass = (
        Vus_provided_N
        >= Vus_required_N
    )

    # --------------------------------------------------------
    # Overall shear
    # --------------------------------------------------------

    overall_pass = (
        maximum_shear_pass
        and shear_reinforcement_capacity_pass
    )

    reasons = []

    if not maximum_shear_pass:

        reasons.append(
            "nominal shear stress exceeds tau_c,max"
        )

    if not shear_reinforcement_capacity_pass:

        reasons.append(
            "provided stirrups do not provide "
            "required shear resistance"
        )

    return {

        "pass": overall_pass,

        "reason": (
            "OK"
            if not reasons
            else "; ".join(reasons)
        ),

        "Vu_kN": Vu_kN,

        "tau_v_MPa": tau_v,

        "tau_c_MPa": tau_c,

        "tau_c_max_MPa": tau_c_max,

        "pt_percent": pt_percent,

        "concrete_shear_pass": (
            concrete_shear_pass
        ),

        "maximum_shear_stress_pass": (
            maximum_shear_pass
        ),

        "shear_reinforcement_required": (
            shear_reinforcement_required
        ),

        "Vus_required_N": (
            Vus_required_N
        ),

        "Vus_provided_N": (
            Vus_provided_N
        ),

        "Asv_mm2": Asv,

        "spacing_required_from_shear_mm": (
            spacing_required_from_shear_mm
        ),

        "shear_reinforcement_capacity_pass": (
            shear_reinforcement_capacity_pass
        ),

        "fy_stirrup_used_MPa": fyv,
    }


# ============================================================
# MINIMUM SHEAR REINFORCEMENT
# ============================================================

def check_minimum_stirrups(
    b_mm: float,
    phi_stirrup_mm: float,
    stirrup_legs: int,
    fy_stirrup_MPa: float,
    stirrup_spacing_mm: float
) -> Dict[str, Any]:

    fyv = min(
        fy_stirrup_MPa,
        415.0
    )

    Asv = (
        stirrup_legs
        * pi
        * phi_stirrup_mm ** 2
        / 4.0
    )

    provided_ratio = (
        Asv
        / (
            b_mm
            * stirrup_spacing_mm
        )
    )

    required_ratio = (
        0.4
        / (
            0.87
            * fyv
        )
    )

    pass_check = (
        provided_ratio
        >= required_ratio
    )

    return {

        "pass": pass_check,

        "reason": (
            "OK"
            if pass_check
            else
            "Minimum shear reinforcement requirement not satisfied."
        ),

        "Asv_mm2": Asv,

        "provided_Asv_over_bs": (
            provided_ratio
        ),

        "required_Asv_over_bs": (
            required_ratio
        ),

        "fy_stirrup_used_MPa": fyv,
    }


# ============================================================
# STIRRUP SPACING
# ============================================================

def check_stirrup_spacing(
    d_mm: float,
    stirrup_spacing_mm: float
) -> Dict[str, Any]:

    allowable_spacing = min(
        0.75 * d_mm,
        300.0
    )

    pass_check = (
        stirrup_spacing_mm
        <= allowable_spacing
    )

    return {

        "pass": pass_check,

        "reason": (
            "OK"
            if pass_check
            else
            "Stirrup spacing exceeds maximum permitted spacing."
        ),

        "provided_spacing_mm": (
            stirrup_spacing_mm
        ),

        "allowable_spacing_mm": (
            allowable_spacing
        ),
    }


# ============================================================
# DEFLECTION
# ============================================================

def check_deflection(
    span_mm: float,
    d_mm: float,
    support_condition: str = "simply_supported",
    modification_factor: float = 1.0
) -> Dict[str, Any]:

    support = (
        support_condition
        .lower()
        .strip()
    )

    if support in [
        "simply_supported",
        "simply supported",
    ]:

        basic_ratio = 20.0

    elif support in [
        "continuous",
        "continuous_beam",
    ]:

        basic_ratio = 26.0

    elif support == "cantilever":

        basic_ratio = 7.0

    else:

        raise ValueError(
            "Unsupported support condition."
        )

    allowable_ratio = (
        basic_ratio
        * modification_factor
    )

    actual_ratio = (
        span_mm
        / d_mm
    )

    pass_check = (
        actual_ratio
        <= allowable_ratio
    )

    return {

        "pass": pass_check,

        "reason": (
            "OK"
            if pass_check
            else
            "Span/depth deflection screening criterion not satisfied."
        ),

        "span_mm": span_mm,

        "effective_depth_mm": d_mm,

        "basic_span_depth_ratio": (
            basic_ratio
        ),

        "modification_factor": (
            modification_factor
        ),

        "allowable_span_depth_ratio": (
            allowable_ratio
        ),

        "actual_span_depth_ratio": (
            actual_ratio
        ),
    }


# ============================================================
# DEVELOPMENT LENGTH
# ============================================================

def calculate_development_length(
    phi_main_mm: float,
    fy_MPa: float,
    fck_MPa: float,
    deformed_bar: bool = True
) -> Dict[str, float]:

    tau_bd = (
        get_tau_bd(
            fck_MPa,
            deformed_bar
        )
    )

    sigma_s = (
        0.87
        * fy_MPa
    )

    Ld = (
        phi_main_mm
        * sigma_s
        / (
            4.0
            * tau_bd
        )
    )

    return {

        "Ld_mm": Ld,

        "tau_bd_MPa": tau_bd,

        "sigma_s_MPa": sigma_s,
    }


def check_development_length(
    available_development_length_mm: float,
    phi_main_mm: float,
    fy_MPa: float,
    fck_MPa: float,
    deformed_bar: bool = True
) -> Dict[str, Any]:

    result = (
        calculate_development_length(
            phi_main_mm,
            fy_MPa,
            fck_MPa,
            deformed_bar
        )
    )

    required_Ld = (
        result["Ld_mm"]
    )

    pass_check = (
        available_development_length_mm
        >= required_Ld
    )

    return {

        "pass": pass_check,

        "reason": (
            "OK"
            if pass_check
            else
            "Available development length is insufficient."
        ),

        "available_length_mm": (
            available_development_length_mm
        ),

        "required_Ld_mm": (
            required_Ld
        ),

        "Ld_over_phi": (
            required_Ld
            / phi_main_mm
        ),

        "tau_bd_MPa": (
            result["tau_bd_MPa"]
        ),

        "sigma_s_MPa": (
            result["sigma_s_MPa"]
        ),
    }


# ============================================================
# REINFORCEMENT DETAILING
# ============================================================

def check_reinforcement_detailing(
    b_mm: float,
    D_mm: float,
    nominal_cover_mm: float,
    phi_main_mm: float,
    phi_stirrup_mm: float,
    n_bars: int,
    bar_center_spacing_mm: float,
    stirrup_spacing_mm: float
) -> Dict[str, Any]:

    minimum_clear_spacing_mm = max(
        phi_main_mm,
        25.0
    )

    clear_spacing_mm = (
        bar_center_spacing_mm
        - phi_main_mm
    )

    minimum_clear_spacing_pass = (
        clear_spacing_mm
        >= minimum_clear_spacing_mm
    )

    maximum_bar_spacing_pass = (
        bar_center_spacing_mm
        <= 300.0
    )

    minimum_main_bar_diameter_pass = (
        phi_main_mm
        >= 12.0
    )

    minimum_stirrup_diameter_pass = (
        phi_stirrup_mm
        >= 8.0
    )

    cover_pass = (
        nominal_cover_mm
        >= 0.0
    )

    number_of_bars_pass = (
        n_bars >= 2
    )

    stirrup_spacing_result = (
        check_stirrup_spacing(
            D_mm
            - nominal_cover_mm
            - phi_stirrup_mm
            - phi_main_mm / 2.0,
            stirrup_spacing_mm
        )
    )

    overall_pass = all([
        minimum_clear_spacing_pass,
        maximum_bar_spacing_pass,
        minimum_main_bar_diameter_pass,
        minimum_stirrup_diameter_pass,
        cover_pass,
        number_of_bars_pass,
        stirrup_spacing_result["pass"],
    ])

    reasons = []

    if not minimum_clear_spacing_pass:

        reasons.append(
            "insufficient clear spacing between main bars"
        )

    if not maximum_bar_spacing_pass:

        reasons.append(
            "main bar spacing exceeds 300 mm"
        )

    if not minimum_main_bar_diameter_pass:

        reasons.append(
            "main bar diameter is less than 12 mm"
        )

    if not minimum_stirrup_diameter_pass:

        reasons.append(
            "stirrup diameter is less than 8 mm"
        )

    if not cover_pass:

        reasons.append(
            "invalid nominal cover"
        )

    if not number_of_bars_pass:

        reasons.append(
            "fewer than two main bars"
        )

    if not stirrup_spacing_result["pass"]:

        reasons.append(
            "stirrup spacing exceeds limit"
        )

    return {

        "pass": overall_pass,

        "reason": (
            "OK"
            if not reasons
            else "; ".join(reasons)
        ),

        "bar_center_spacing_mm": (
            bar_center_spacing_mm
        ),

        "bar_clear_spacing_mm": (
            clear_spacing_mm
        ),

        "minimum_clear_spacing_mm": (
            minimum_clear_spacing_mm
        ),

        "minimum_clear_spacing_pass": (
            minimum_clear_spacing_pass
        ),

        "maximum_bar_spacing_pass": (
            maximum_bar_spacing_pass
        ),

        "minimum_main_bar_diameter_pass": (
            minimum_main_bar_diameter_pass
        ),

        "minimum_stirrup_diameter_pass": (
            minimum_stirrup_diameter_pass
        ),

        "cover_pass": cover_pass,

        "number_of_bars_pass": (
            number_of_bars_pass
        ),

        "stirrup_spacing": (
            stirrup_spacing_result
        ),
    }


# ============================================================
# COMPLETE BEAM CHECK
# ============================================================

def check_beam(
    ADL_kNm: float,
    LL_kNm: float,
    span_mm: float,
    b_mm: float,
    D_mm: float,
    fck_MPa: float,
    fy_MPa: float,
    Ast_prov_mm2: float,
    n_bars: int,
    phi_main_mm: float,
    phi_stirrup_mm: float,
    stirrup_legs: int = 2,
    stirrup_spacing_mm: float = 200.0,
    nominal_cover_mm: float = 30.0,
    fy_stirrup_MPa: float = 415.0,
    support_condition: str = "simply_supported",
    available_development_length_mm: float = None,
    dead_load_factor: float = 1.5,
    live_load_factor: float = 1.5,
    concrete_density_kN_m3: float = 25.0,
    deflection_modification_factor: float = 1.0
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # INPUT VALIDATION
    # --------------------------------------------------------

    validate_materials(
        fck_MPa,
        fy_MPa
    )

    validate_positive(
        span_mm,
        "span_mm"
    )

    validate_positive(
        b_mm,
        "b_mm"
    )

    validate_positive(
        D_mm,
        "D_mm"
    )

    validate_positive(
        Ast_prov_mm2,
        "Ast_prov_mm2"
    )

    validate_positive(
        phi_main_mm,
        "phi_main_mm"
    )

    validate_positive(
        phi_stirrup_mm,
        "phi_stirrup_mm"
    )

    if n_bars < 2:

        raise ValueError(
            "n_bars must be at least 2."
        )

    if stirrup_legs < 2:

        raise ValueError(
            "stirrup_legs must be at least 2."
        )

    # --------------------------------------------------------
    # EFFECTIVE DEPTH
    # --------------------------------------------------------

    d_mm = (
        calculate_effective_depth(
            D_mm,
            nominal_cover_mm,
            phi_main_mm
        )
    )

    if d_mm <= 0:

        return {
            "feasible": False,

            "failure_reasons": [
                "Effective depth is non-positive."
            ],
        }

    # --------------------------------------------------------
    # LONGITUDINAL BAR GEOMETRY
    # --------------------------------------------------------

    spacing = (
        calculate_longitudinal_bar_spacing(
            b_mm=b_mm,
            nominal_cover_mm=nominal_cover_mm,
            phi_stirrup_mm=phi_stirrup_mm,
            phi_main_mm=phi_main_mm,
            n_bars=n_bars
        )
    )

    bar_center_spacing_mm = (
        spacing["bar_center_spacing_mm"]
    )

    # --------------------------------------------------------
    # SELF WEIGHT
    # --------------------------------------------------------

    self_weight_kNm = (
        calculate_self_weight(
            b_mm,
            D_mm,
            concrete_density_kN_m3
        )
    )

    # --------------------------------------------------------
    # DESIGN LOAD
    # --------------------------------------------------------

    loads = (
        calculate_design_loads(
            ADL_kNm=ADL_kNm,
            LL_kNm=LL_kNm,
            self_weight_kNm=self_weight_kNm,
            dead_load_factor=dead_load_factor,
            live_load_factor=live_load_factor
        )
    )

    # --------------------------------------------------------
    # DESIGN ACTIONS
    # --------------------------------------------------------

    actions = (
        calculate_simple_span_actions(
            loads["factored_load_kNm"],
            span_mm
        )
    )

    Mu_kNm = (
        actions["Mu_kNm"]
    )

    Vu_kN = (
        actions["Vu_kN"]
    )

    # --------------------------------------------------------
    # FLEXURE
    # --------------------------------------------------------

    flexure = (
        check_flexure(
            b_mm=b_mm,
            D_mm=D_mm,
            d_mm=d_mm,
            Ast_prov_mm2=Ast_prov_mm2,
            fck_MPa=fck_MPa,
            fy_MPa=fy_MPa,
            Mu_kNm=Mu_kNm
        )
    )

    # --------------------------------------------------------
    # SHEAR
    # --------------------------------------------------------

    shear = (
        check_shear(
            b_mm=b_mm,
            d_mm=d_mm,
            Ast_prov_mm2=Ast_prov_mm2,
            fck_MPa=fck_MPa,
            Vu_kN=Vu_kN,
            phi_stirrup_mm=phi_stirrup_mm,
            stirrup_legs=stirrup_legs,
            fy_stirrup_MPa=fy_stirrup_MPa,
            stirrup_spacing_mm=stirrup_spacing_mm
        )
    )

    # --------------------------------------------------------
    # MINIMUM STIRRUPS
    # --------------------------------------------------------

    minimum_stirrups = (
        check_minimum_stirrups(
            b_mm=b_mm,
            phi_stirrup_mm=phi_stirrup_mm,
            stirrup_legs=stirrup_legs,
            fy_stirrup_MPa=fy_stirrup_MPa,
            stirrup_spacing_mm=stirrup_spacing_mm
        )
    )

    # --------------------------------------------------------
    # STIRRUP SPACING
    # --------------------------------------------------------

    stirrup_spacing = (
        check_stirrup_spacing(
            d_mm=d_mm,
            stirrup_spacing_mm=stirrup_spacing_mm
        )
    )

    # --------------------------------------------------------
    # DEFLECTION
    # --------------------------------------------------------

    deflection = (
        check_deflection(
            span_mm=span_mm,
            d_mm=d_mm,
            support_condition=support_condition,
            modification_factor=(
                deflection_modification_factor
            )
        )
    )

    # --------------------------------------------------------
    # DEVELOPMENT LENGTH
    # --------------------------------------------------------

    if available_development_length_mm is None:

        # Conservative project-screening assumption:
        # full span is available.
        #
        # Candidate generation should preferably provide
        # an explicit anchorage/development length.
        available_development_length_mm = span_mm

    development = (
        check_development_length(
            available_development_length_mm=(
                available_development_length_mm
            ),
            phi_main_mm=phi_main_mm,
            fy_MPa=fy_MPa,
            fck_MPa=fck_MPa,
            deformed_bar=True
        )
    )

    # --------------------------------------------------------
    # REINFORCEMENT DETAILING
    # --------------------------------------------------------

    detailing = (
        check_reinforcement_detailing(
            b_mm=b_mm,
            D_mm=D_mm,
            nominal_cover_mm=nominal_cover_mm,
            phi_main_mm=phi_main_mm,
            phi_stirrup_mm=phi_stirrup_mm,
            n_bars=n_bars,
            bar_center_spacing_mm=(
                bar_center_spacing_mm
            ),
            stirrup_spacing_mm=(
                stirrup_spacing_mm
            )
        )
    )

    # --------------------------------------------------------
    # FAILURE REASONS
    # --------------------------------------------------------

    failure_reasons = []

    if not flexure["pass"]:

        failure_reasons.append(
            "Flexure"
        )

    if not shear["pass"]:

        failure_reasons.append(
            "Shear"
        )

    if not minimum_stirrups["pass"]:

        failure_reasons.append(
            "Minimum shear reinforcement"
        )

    if not stirrup_spacing["pass"]:

        failure_reasons.append(
            "Stirrup spacing"
        )

    if not deflection["pass"]:

        failure_reasons.append(
            "Deflection"
        )

    if not development["pass"]:

        failure_reasons.append(
            "Development length"
        )

    if not detailing["pass"]:

        failure_reasons.append(
            "Reinforcement detailing"
        )

    feasible = (
        len(failure_reasons) == 0
    )

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {

        "feasible": feasible,

        "failure_reasons": (
            failure_reasons
        ),

        "geometry": {

            "b_mm": b_mm,

            "D_mm": D_mm,

            "d_mm": d_mm,

            "nominal_cover_mm": (
                nominal_cover_mm
            ),
        },

        "materials": {

            "fck_MPa": fck_MPa,

            "fy_MPa": fy_MPa,

            "fy_stirrup_MPa": (
                fy_stirrup_MPa
            ),
        },

        "reinforcement": {

            "Ast_prov_mm2": (
                Ast_prov_mm2
            ),

            "n_bars": n_bars,

            "phi_main_mm": (
                phi_main_mm
            ),

            "phi_stirrup_mm": (
                phi_stirrup_mm
            ),

            "stirrup_legs": (
                stirrup_legs
            ),

            "stirrup_spacing_mm": (
                stirrup_spacing_mm
            ),

            "bar_center_spacing_mm": (
                bar_center_spacing_mm
            ),

            "bar_clear_spacing_mm": (
                spacing["bar_clear_spacing_mm"]
            ),
        },

        "loads": {

            "ADL_kNm": ADL_kNm,

            "LL_kNm": LL_kNm,

            "self_weight_kNm": (
                loads["self_weight_kNm"]
            ),

            "total_dead_load_kNm": (
                loads["total_dead_load_kNm"]
            ),

            "factored_load_kNm": (
                loads["factored_load_kNm"]
            ),
        },

        "actions": {

            "Mu_kNm": Mu_kNm,

            "Vu_kN": Vu_kN,
        },

        "flexure": flexure,

        "shear": shear,

        "minimum_stirrups": (
            minimum_stirrups
        ),

        "stirrup_spacing": (
            stirrup_spacing
        ),

        "deflection": deflection,

        "development_length": (
            development
        ),

        "detailing": detailing,
    }


# ============================================================
# REPORT
# ============================================================

def print_check_report(
    result: Dict[str, Any]
) -> None:

    print("\n" + "=" * 70)
    print("RCC BEAM STRUCTURAL CHECK REPORT")
    print("=" * 70)

    print(
        "\nOverall status: "
        + (
            "PASS"
            if result["feasible"]
            else "FAIL"
        )
    )

    geometry = result["geometry"]

    print("\nGeometry")

    print(
        f"  Width b       : "
        f"{geometry['b_mm']:.1f} mm"
    )

    print(
        f"  Depth D       : "
        f"{geometry['D_mm']:.1f} mm"
    )

    print(
        f"  Effective d   : "
        f"{geometry['d_mm']:.1f} mm"
    )

    materials = result["materials"]

    print("\nMaterials")

    print(
        f"  Concrete fck  : "
        f"{materials['fck_MPa']:.1f} MPa"
    )

    print(
        f"  Steel fy      : "
        f"{materials['fy_MPa']:.1f} MPa"
    )

    loads = result["loads"]

    print("\nLoads")

    print(
        f"  ADL           : "
        f"{loads['ADL_kNm']:.3f} kN/m"
    )

    print(
        f"  LL            : "
        f"{loads['LL_kNm']:.3f} kN/m"
    )

    print(
        f"  Self weight   : "
        f"{loads['self_weight_kNm']:.3f} kN/m"
    )

    print(
        f"  Factored load : "
        f"{loads['factored_load_kNm']:.3f} kN/m"
    )

    actions = result["actions"]

    print("\nDesign Actions")

    print(
        f"  Mu            : "
        f"{actions['Mu_kNm']:.3f} kN-m"
    )

    print(
        f"  Vu            : "
        f"{actions['Vu_kN']:.3f} kN"
    )

    flexure = result["flexure"]

    print("\nFlexure")

    print(
        f"  Demand              : "
        f"{flexure['Mu_demand_kNm']:.3f} kN-m"
    )

    print(
        f"  Limiting capacity   : "
        f"{flexure['Mu_lim_kNm']:.3f} kN-m"
    )

    print(
        f"  Utilization         : "
        f"{flexure['flexural_utilization']:.3f}"
    )

    print(
        f"  xu                  : "
        f"{flexure['xu_mm']:.2f} mm"
    )

    print(
        f"  xu,max              : "
        f"{flexure['xu_max_mm']:.2f} mm"
    )

    print(
        f"  Ast provided        : "
        f"{flexure['Ast_prov_mm2']:.2f} mm2"
    )

    print(
        f"  Ast minimum        : "
        f"{flexure['Ast_min_mm2']:.2f} mm2"
    )

    print(
        f"  Ast limiting       : "
        f"{flexure['Ast_lim_mm2']:.2f} mm2"
    )

    print(
        f"  Ast maximum        : "
        f"{flexure['Ast_max_mm2']:.2f} mm2"
    )

    print(
        f"  Neutral-axis check  : "
        f"{'PASS' if flexure['neutral_axis_pass'] else 'FAIL'}"
    )

    print(
        f"  Limiting capacity   : "
        f"{'PASS' if flexure['limiting_capacity_pass'] else 'FAIL'}"
    )

    print(
        f"  Minimum steel       : "
        f"{'PASS' if flexure['minimum_steel_pass'] else 'FAIL'}"
    )

    print(
        f"  Maximum steel       : "
        f"{'PASS' if flexure['maximum_steel_pass'] else 'FAIL'}"
    )

    print(
        f"  Overall flexure     : "
        f"{'PASS' if flexure['pass'] else 'FAIL'}"
    )

    if not flexure["pass"]:

        print(
            f"  Reason              : "
            f"{flexure['reason']}"
        )

    shear = result["shear"]

    print("\nShear")

    print(
        f"  tau_v               : "
        f"{shear['tau_v_MPa']:.3f} MPa"
    )

    print(
        f"  tau_c               : "
        f"{shear['tau_c_MPa']:.3f} MPa"
    )

    print(
        f"  tau_c,max           : "
        f"{shear['tau_c_max_MPa']:.3f} MPa"
    )

    print(
        f"  Required Vus        : "
        f"{shear['Vus_required_N'] / 1000.0:.3f} kN"
    )

    print(
        f"  Provided Vus        : "
        f"{shear['Vus_provided_N'] / 1000.0:.3f} kN"
    )

    print(
        f"  Shear stress check  : "
        f"{'PASS' if shear['maximum_shear_stress_pass'] else 'FAIL'}"
    )

    print(
        f"  Stirrup capacity    : "
        f"{'PASS' if shear['shear_reinforcement_capacity_pass'] else 'FAIL'}"
    )

    print(
        f"  Overall shear       : "
        f"{'PASS' if shear['pass'] else 'FAIL'}"
    )

    minimum_stirrups = (
        result["minimum_stirrups"]
    )

    print("\nMinimum Shear Reinforcement")

    print(
        f"  Provided Asv/(bs)   : "
        f"{minimum_stirrups['provided_Asv_over_bs']:.6f}"
    )

    print(
        f"  Required Asv/(bs)   : "
        f"{minimum_stirrups['required_Asv_over_bs']:.6f}"
    )

    print(
        f"  Status              : "
        f"{'PASS' if minimum_stirrups['pass'] else 'FAIL'}"
    )

    stirrup_spacing = (
        result["stirrup_spacing"]
    )

    print("\nStirrup Spacing")

    print(
        f"  Provided spacing    : "
        f"{stirrup_spacing['provided_spacing_mm']:.1f} mm"
    )

    print(
        f"  Allowable spacing   : "
        f"{stirrup_spacing['allowable_spacing_mm']:.1f} mm"
    )

    print(
        f"  Status              : "
        f"{'PASS' if stirrup_spacing['pass'] else 'FAIL'}"
    )

    deflection = result["deflection"]

    print("\nDeflection")

    print(
        f"  Actual L/d          : "
        f"{deflection['actual_span_depth_ratio']:.2f}"
    )

    print(
        f"  Allowable L/d       : "
        f"{deflection['allowable_span_depth_ratio']:.2f}"
    )

    print(
        f"  Status              : "
        f"{'PASS' if deflection['pass'] else 'FAIL'}"
    )

    development = (
        result["development_length"]
    )

    print("\nDevelopment Length")

    print(
        f"  Required Ld         : "
        f"{development['required_Ld_mm']:.1f} mm"
    )

    print(
        f"  Available           : "
        f"{development['available_length_mm']:.1f} mm"
    )

    print(
        f"  Ld / phi            : "
        f"{development['Ld_over_phi']:.2f}"
    )

    print(
        f"  Status              : "
        f"{'PASS' if development['pass'] else 'FAIL'}"
    )

    detailing = result["detailing"]

    print("\nReinforcement Detailing")

    print(
        f"  Bar spacing         : "
        f"{detailing['bar_center_spacing_mm']:.1f} mm"
    )

    print(
        f"  Clear bar spacing   : "
        f"{detailing['bar_clear_spacing_mm']:.1f} mm"
    )

    print(
        f"  Minimum clear gap   : "
        f"{'PASS' if detailing['minimum_clear_spacing_pass'] else 'FAIL'}"
    )

    print(
        f"  Maximum bar spacing : "
        f"{'PASS' if detailing['maximum_bar_spacing_pass'] else 'FAIL'}"
    )

    print(
        f"  Main bar diameter   : "
        f"{'PASS' if detailing['minimum_main_bar_diameter_pass'] else 'FAIL'}"
    )

    print(
        f"  Stirrup diameter    : "
        f"{'PASS' if detailing['minimum_stirrup_diameter_pass'] else 'FAIL'}"
    )

    print(
        f"  Cover               : "
        f"{'PASS' if detailing['cover_pass'] else 'FAIL'}"
    )

    print(
        f"  Number of bars      : "
        f"{'PASS' if detailing['number_of_bars_pass'] else 'FAIL'}"
    )

    print(
        f"  Overall detailing   : "
        f"{'PASS' if detailing['pass'] else 'FAIL'}"
    )

    if result["failure_reasons"]:

        print("\nFailure reasons:")

        for reason in result["failure_reasons"]:

            print(
                f"  - {reason}"
            )

    else:

        print(
            "\nAll implemented checks passed."
        )

    print("=" * 70)


# ============================================================
# TEST CASE
# ============================================================

def run_example():

    result = check_beam(

        ADL_kNm=5.0,

        LL_kNm=5.0,

        span_mm=5000.0,

        b_mm=250.0,

        D_mm=450.0,

        fck_MPa=25.0,

        fy_MPa=500.0,

        Ast_prov_mm2=1570.0,

        n_bars=4,

        phi_main_mm=16.0,

        phi_stirrup_mm=8.0,

        stirrup_legs=2,

        stirrup_spacing_mm=200.0,

        nominal_cover_mm=30.0,

        fy_stirrup_MPa=415.0,

        support_condition="simply_supported",

        available_development_length_mm=5000.0,

        dead_load_factor=1.5,

        live_load_factor=1.5,

        concrete_density_kN_m3=25.0,

        deflection_modification_factor=1.0,
    )

    print_check_report(
        result
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_example()