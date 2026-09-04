from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

"""
Logarithmic-spiral maths for the SpiRob geometry.

Pure, testable functions with no global state, plus a small parameter
container and a factory for root finding.
"""

__all__ = [
    "rho",
    "rho_c",
    "length_central",
    "delta_width",
    "theta0_from_ratio",
    "a_from_tip",
    "L_from_b",
    "f_of_b",
    "taper_angle_phi",
    "SpiralParams",
    "make_f_of_b",
    "distance_point_line_from_origin",
    "closest_point_on_line_to_origin",
    "line_unit_direction",
    "circle_radius_for_central_angle",
    "intersection_points_circle_line",
    "angle_between_rays",
    "solve_for_points",
]


def rho(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Radius of the logarithmic spiral.

    The spiral is defined by
    :math:`\\rho(\\theta) = a \\cdot e^{b\\,\\theta}`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polar angle :math:`\\theta` in radians. Scalar or array.
    a : float
        Scale factor :math:`a` (normally > 0).
    b : float
        Growth parameter :math:`b` (positive or negative; not 0 for
        some downstream computations; this function itself tolerates 0).

    Returns
    -------
    float or np.ndarray
        Radius :math:`\\rho(\\theta)`; shape follows ``theta``.

    Notes
    -----
    - For :math:`b=0`, :math:`\\rho(\\theta)=a` (a circle).
    - Vectorized over ``numpy`` arrays.

    Examples
    --------
    >>> import numpy as np
    >>> rho(0.0, a=1.0, b=0.2)
    1.0
    >>> th = np.array([0.0, np.pi])
    >>> rho(th, a=1.0, b=0.2).shape == th.shape
    True
    """
    return a * np.exp(b * theta)


def rho_c(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Helper quantity :math:`\\rho_c(\\theta)` used for the length integral.

    Defined as
    :math:`\\rho_c(\\theta) = \\tfrac{1}{2} a\\,(e^{2\\pi b}+1)\\,e^{b\\theta}`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polar angle in radians. Scalar or array.
    a : float
        Scale factor of the spiral.
    b : float
        Wachstumsparameter.

    Returns
    -------
    float or np.ndarray
        :math:`\\rho_c(\\theta)`; shape follows ``theta``.

    Notes
    -----
    It appears in the closed-form integral of the centreline length.
    """
    return 0.5 * a * (np.exp(2 * np.pi * b) + 1.0) * np.exp(b * theta)


def length_central(a: float, b: float, theta0: float) -> float:
    """
    Centreline length from :math:`\\theta=0` to :math:`\\theta=\\theta_0`.

    Uses the closed form
    :math:`L = \\frac{\\sqrt{1+b^2}}{b}\\,[\\rho_c(\\theta_0)-\\rho_c(0)]`.

    Parameters
    ----------
    a : float
        Scale factor of the spiral (normally > 0).
    b : float
        Growth parameter; **must not be 0** (division by zero).
    theta0 : float
        End angle in radians (>= 0 recommended).

    Returns
    -------
    float
        Centreline length :math:`L(0\\to\\theta_0)`.

    Raises
    ------
    ValueError
        If ``b`` is near 0 (division by zero).

    Notes
    -----
    For very small |b| the formula is poorly conditioned.

    Examples
    --------
    >>> length_central(a=0.01, b=0.2, theta0=np.pi) > 0
    True
    """
    if np.isclose(b, 0.0):
        raise ValueError("length_central: b must not be 0 (division by zero).")
    return (np.sqrt(b**2 + 1.0) / b) * (rho_c(theta0, a, b) - rho_c(0.0, a, b))


def delta_width(theta: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    """
    Difference between two radial spirals after one full turn.

    Defined as
    :math:`\\delta(\\theta) = a\\,(e^{b(\\theta+2\\pi)} - e^{b\\theta})`.

    Parameters
    ----------
    theta : float or np.ndarray
        Polar angle in radians.
    a : float
        Skalenfaktor.
    b : float
        Wachstumsparameter.

    Returns
    -------
    float or np.ndarray
        :math:`\\delta(\\theta)`; shape follows ``theta``.

    Examples
    --------
    >>> delta_width(0.0, a=1.0, b=0.2) > 0
    True
    """
    return a * (np.exp(b * (theta + 2 * np.pi)) - np.exp(b * theta))


def theta0_from_ratio(b: float, base_d: float, tip_d: float) -> float:
    """
    Compute :math:`\\theta_0` from the base/tip diameter ratio.

    Formel:
    :math:`\\theta_0 = \\frac{\\ln(\\tfrac{\\text{base\\_d}}{\\text{tip\\_d}})}{b}`.

    Parameters
    ----------
    b : float
        Growth parameter; **must not be 0**.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        :math:`\\theta_0` (radians).

    Raises
    ------
    ValueError
        If ``b`` is near 0 or a diameter is not > 0.

    Examples
    --------
    >>> theta0_from_ratio(0.2, base_d=0.06, tip_d=0.01) > 0
    True
    """
    if base_d <= 0 or tip_d <= 0:
        raise ValueError("theta0_from_ratio: base_d and tip_d must be > 0.")
    if np.isclose(b, 0.0):
        raise ValueError("theta0_from_ratio: b must not be 0.")
    return np.log(base_d / tip_d) / b


def a_from_tip(b: float, tip_d: float) -> float:
    """
    Determine :math:`a` from the tip diameter :math:`\\text{tip\\_d}`.

    Formel:
    :math:`a = \\dfrac{\\text{tip\\_d}}{e^{2\\pi b}-1}`.

    Parameters
    ----------
    b : float
        Wachstumsparameter.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        Skalenfaktor :math:`a`.

    Raises
    ------
    ValueError
        If ``tip_d`` is not > 0, or the denominator is numerically 0
        (typically when :math:`b \\approx 0`).

    Examples
    --------
    >>> a_from_tip(0.2, tip_d=0.01) > 0
    True
    """
    if tip_d <= 0:
        raise ValueError("a_from_tip: tip_d must be > 0.")
    denom = np.exp(2 * np.pi * b) - 1.0
    if np.isclose(denom, 0.0):
        raise ValueError("a_from_tip: exp(2*pi*b) - 1 is 0 (b is approximately 0).")
    return tip_d / denom


def L_from_b(b: float, base_d: float, tip_d: float) -> float:
    """
    Centreline length :math:`L(b)` for given base/tip diameters.

    Intern:
    1) :math:`\\theta_0 = \\ln(base\\_d/tip\\_d)/b`
    2) :math:`a = \\text{tip\\_d}/(e^{2\\pi b}-1)`
    3) :math:`L = \\text{length\\_central}(a,b,\\theta_0)`

    Parameters
    ----------
    b : float
        Growth parameter; must not be 0.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.

    Returns
    -------
    float
        Centreline length :math:`L(b)`.

    Raises
    ------
    ValueError
        On invalid parameters (see the called functions).

    Examples
    --------
    >>> L_from_b(0.2, base_d=0.06, tip_d=0.01) > 0
    True
    """
    th0 = theta0_from_ratio(b, base_d, tip_d)
    a_b = a_from_tip(b, tip_d)
    return length_central(a_b, b, th0)


def f_of_b(b: float, base_d: float, tip_d: float, L_target: float) -> float:
    """
    Objective :math:`f(b) = L(b) - L_{target}` for root finding.

    Parameters
    ----------
    b : float
        Wachstumsparameter.
    base_d : float
        Basisdurchmesser > 0.
    tip_d : float
        Spitzendurchmesser > 0.
    L_target : float
        Target centreline length, > 0.

    Returns
    -------
    float
        The difference :math:`f(b)`; its root satisfies the length target.

    Raises
    ------
    ValueError
        If ``L_target`` is not > 0.

    Examples
    --------
    >>> f_of_b(0.2, base_d=0.06, tip_d=0.01, L_target=0.25)  # doctest:+ELLIPSIS
    ...
    """
    if L_target <= 0:
        raise ValueError("f_of_b: L_target must be > 0.")
    return L_from_b(b, base_d, tip_d) - L_target


def taper_angle_phi(b: float) -> float:
    """
    Taper angle :math:`\\varphi` (in radians).

    Formel:
    :math:`\\varphi = 2\\,\\arctan\\!\\left(\\dfrac{b\\,(e^{2\\pi b}-1)}{\\sqrt{1+b^2}\\,(e^{2\\pi b}+1)}\\right)`.

    Parameters
    ----------
    b : float
        Wachstumsparameter (reell).

    Returns
    -------
    float
        :math:`\\varphi` in radians.

    Notes
    -----
    - For small :math:`|b|` the angle is small as well.
    - The expression is numerically stable over the usual working range.

    Examples
    --------
    >>> 0.0 <= taper_angle_phi(0.2) <= np.pi
    True
    """
    num = b * (np.exp(2 * np.pi * b) - 1.0)
    den = np.sqrt(b**2 + 1.0) * (np.exp(2 * np.pi * b) + 1.0)
    return 2.0 * np.arctan(num / den)



@dataclass(frozen=True)
class SpiralParams:
    """
    Container for the parameters that are usually passed together.

    Attributes
    ----------
    base_d : float
        Basisdurchmesser (> 0).
    tip_d : float
        Spitzendurchmesser (> 0).
    L_target : float
        Target centreline length (> 0).
    """
    base_d: float
    tip_d: float
    L_target: float

    def validate(self) -> None:
        """
        Validate the parameters.

        Raises
        ------
        ValueError
            If any value is not > 0.
        """
        if self.base_d <= 0 or self.tip_d <= 0 or self.L_target <= 0:
            raise ValueError(
                "SpiralParams: base_d, tip_d and L_target must be > 0."
            )


def make_f_of_b(params: SpiralParams) -> Callable[[float], float]:
    """
    Build a one-dimensional function :math:`f(b) = L(b) - L_{target}`.

    Convenient for root finding (e.g. ``scipy.optimize.brentq``).

    Parameters
    ----------
    params : SpiralParams
        Valid parameters (they are validated).

    Returns
    -------
    Callable[[float], float]
        A function taking only :math:`b`.

    Raises
    ------
    ValueError
        If the parameters are invalid.

    Examples
    --------
    >>> from math import isfinite
    >>> p = SpiralParams(base_d=0.06, tip_d=0.01, L_target=0.25)
    >>> f = make_f_of_b(p)  # doctest:+ELLIPSIS
    >>> isfinite(float(f(0.2)))
    True
    """
    params.validate()

    def _f(b: float) -> float:
        return f_of_b(b, base_d=params.base_d, tip_d=params.tip_d, L_target=params.L_target)

    return _f



def distance_point_line_from_origin(a: float, b: float, c: float) -> float:
    """
    Distance from the origin (0,0) to the line :math:`a x + b y + c = 0`.

    Parameters
    ----------
    a, b, c : float
        Coefficients of the line in normal form.

    Returns
    -------
    float
        Abstand :math:`d = |c| / \\sqrt{a^2 + b^2}`.

    Raises
    ------
    ValueError
        If ``a = b = 0`` (not a valid line).

    Examples
    --------
    >>> distance_point_line_from_origin(0.0, 1.0, -2.0)  # y - 2 = 0
    2.0
    """
    norm = math.hypot(a, b)
    if norm == 0.0:
        raise ValueError("Invalid line: a=b=0.")
    return abs(c) / norm


def closest_point_on_line_to_origin(a: float, b: float, c: float) -> tuple[float, float]:
    """
    Foot of the perpendicular from the origin onto :math:`a x + b y + c = 0`.

    Formel
    ------
    :math:`M = -\\dfrac{c}{a^2 + b^2}\\,(a, b)`.

    Parameters
    ----------
    a, b, c : float
        Coefficients of the line in normal form.

    Returns
    -------
    tuple[float, float]
        Koordinaten :math:`M=(x_M, y_M)`.

    Raises
    ------
    ValueError
        If ``a = b = 0`` (not a valid line).

    Examples
    --------
    >>> closest_point_on_line_to_origin(0.0, 1.0, -2.0)  # y - 2 = 0
    (0.0, 2.0)
    """
    denom = a * a + b * b
    if denom == 0.0:
        raise ValueError("Invalid line: a=b=0.")
    return (-c * a / denom, -c * b / denom)


def line_unit_direction(a: float, b: float) -> tuple[float, float]:
    """
    Unit direction vector of the line :math:`a x + b y + c = 0`.

    Notes
    --------
    A normal vector is :math:`n=(a,b)`; a (right-handed) direction vector is
    :math:`u_\\perp = (-b, a)`. The returned vector is normalised.

    Parameters
    ----------
    a, b : float
        Komponenten des Normalenvektors.

    Returns
    -------
    tuple[float, float]
        Unit direction vector :math:`\\hat{u}` along the line.

    Raises
    ------
    ValueError
        If ``a = b = 0`` (the direction is undefined).

    Examples
    --------
    >>> line_unit_direction(0.0, 1.0)  # line parallel to the x-axis
    (-1.0, 0.0)
    """
    norm = math.hypot(a, b)
    if norm == 0.0:
        raise ValueError("Invalid line: a=b=0.")
    return (-b / norm, a / norm)


def circle_radius_for_central_angle(d: float, theta_deg: float) -> float:
    """
    Circle radius from distance :math:`d` and central angle :math:`\\theta` (degrees).

    Formel
    ------
    :math:`r = \\dfrac{d}{\\cos(\\theta/2)}`  with :math:`\\theta` in degrees.

    Parameters
    ----------
    d : float
        Distance of that perpendicular foot from the origin.
    theta_deg : float
        Central angle in degrees.

    Returns
    -------
    float
        Kreisradius :math:`r`.

    Raises
    ------
    ValueError
        Falls :math:`\\cos(\\theta/2) \\le 0`, d. h. :math:`\\theta/2 \\ge 90^\\circ`.

    Examples
    --------
    >>> circle_radius_for_central_angle(1.0, 60.0)
    1.154700538...
    """
    theta_rad = math.radians(theta_deg)
    c = math.cos(theta_rad / 2.0)
    if c <= 0.0:
        raise ValueError("theta/2 must be < 90 deg (cos > 0).")
    return d / c


def intersection_points_circle_line(
    a: float, b: float, c: float, r: float
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Intersections of the circle :math:`x^2 + y^2 = r^2` with the line :math:`a x + b y + c = 0`.

    Geometrische Konstruktion
    -------------------------
    - :math:`M` is the perpendicular foot from the origin onto the line.
    - :math:`\\hat{u}` is the unit direction vector of the line.
    - :math:`d` is the distance from the origin to the line.
    - :math:`s = \\sqrt{r^2 - d^2}` (non-negative; zero for a tangent).
    - Schnittpunkte: :math:`C = M + s\\,\\hat{u}`, :math:`D = M - s\\,\\hat{u}`.

    Parameters
    ----------
    a, b, c : float
        Coefficients of the line in normal form.
    r : float
        Kreisradius (>= 0).

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Points :math:`C` and :math:`D` as coordinate pairs.

    Raises
    ------
    ValueError
        If the line lies entirely outside the circle (no intersection).

    Examples
    --------
    >>> intersection_points_circle_line(0.0, 1.0, 0.0, 1.0)  # x^2+y^2=1 and y=0
    ((1.0, 0.0), (-1.0, 0.0))
    """
    d = distance_point_line_from_origin(a, b, c)
    if d > r + 1e-12:
        raise ValueError("No intersection: the line lies outside the circle.")
    Mx, My = closest_point_on_line_to_origin(a, b, c)
    ux, uy = line_unit_direction(a, b)
    s_sq = max(r * r - d * d, 0.0)  # numerisch nichtnegativ
    s = math.sqrt(s_sq)
    C = (Mx + s * ux, My + s * uy)
    D = (Mx - s * ux, My - s * uy)
    return C, D


def angle_between_rays(p: tuple[float, float], q: tuple[float, float]) -> float:
    """
    Angle between the rays :math:`OP` and :math:`OQ` (in degrees).

    Parameters
    ----------
    p, q : tuple[float, float]
        End points of the vectors :math:`\\vec{OP}` and :math:`\\vec{OQ}`.

    Returns
    -------
    float
        Angle :math:`\\angle(\\vec{OP},\\vec{OQ})` in degrees, within :math:`[0,180]`.

    Raises
    ------
    ValueError
        If either point lies at the origin (the angle is undefined).

    Examples
    --------
    >>> angle_between_rays((1, 0), (0, 1))
    90.0
    """
    px, py = p
    qx, qy = q
    dp = px * qx + py * qy
    np_ = math.hypot(px, py)
    nq_ = math.hypot(qx, qy)
    if np_ == 0.0 or nq_ == 0.0:
        raise ValueError("Point lies at the origin; the angle is undefined.")
    cosang = dp / (np_ * nq_)
    # numerisch einklammern
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang))


def solve_for_points(
    a: float, b: float, c: float, theta_deg: float = 30.0
) -> dict[str, float | tuple[float, float]]:
    """
    Main entry point: build the circle and intersect it with the given line.

    Ablauf
    ------
    1. Distance :math:`d` from the origin to the line.
    2. Radius :math:`r` from the desired central angle ``theta_deg``.
    3. Intersections :math:`C, D` of the circle and the line.
    4. Validierung: Zwischenwinkel :math:`\\angle COD` (sollte nahe ``theta_deg`` sein).

    Parameters
    ----------
    a, b, c : float
        Coefficients of the line in normal form :math:`a x + b y + c = 0`.
    theta_deg : float, default 30.0
        Central angle in degrees the circle is constructed for.

    Returns
    -------
    dict[str, float | tuple[float, float]]
        Dictionary with keys:
        - ``"d"`` : distance from the origin to the line
        - ``"r"`` : Kreisradius
        - ``"C"`` : erster Schnittpunkt (x, y)
        - ``"D"`` : zweiter Schnittpunkt (x, y)
        - ``"winkel_deg"`` : angle :math:`\\angle COD` in degrees

    Raises
    ------
    ValueError
        On geometrically impossible configurations (e.g. no intersection).

    Examples
    --------
    >>> res = solve_for_points(0.0, 1.0, -0.5, theta_deg=60.0)  # y-0.5=0
    >>> sorted(res.keys())
    ['C', 'D', 'd', 'r', 'winkel_deg']
    """
    d = distance_point_line_from_origin(a, b, c)
    r = circle_radius_for_central_angle(d, theta_deg)
    C, D = intersection_points_circle_line(a, b, c, r)
    w = angle_between_rays(C, D)
    return {"d": d, "r": r, "C": C, "D": D, "winkel_deg": w}



