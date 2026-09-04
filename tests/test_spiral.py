import math

import numpy as np
import pytest

# Pure spiral maths -- no MuJoCo needed.
from spirob.spiral import (
    L_from_b,
    SpiralParams,
    a_from_tip,
    angle_between_rays,
    circle_radius_for_central_angle,
    closest_point_on_line_to_origin,
    # geometry functions
    distance_point_line_from_origin,
    f_of_b,
    intersection_points_circle_line,
    length_central,
    line_unit_direction,
    make_f_of_b,
    # spiral functions
    rho,
    rho_c,
    solve_for_points,
    taper_angle_phi,
    theta0_from_ratio,
)


def test_rho_scalar_and_array():
    assert np.isclose(rho(0.0, a=1.0, b=0.2), 1.0)
    th = np.array([0.0, np.pi])
    out = rho(th, a=1.0, b=0.2)
    assert out.shape == th.shape
    assert np.isclose(out[0], 1.0)

def test_rho_c_scalar():
    assert np.isclose(rho_c(0, 1, 0.1), 0.5 * 1 * (np.exp(2*np.pi*0.1)+1))

def test_rho_c_array():
    th = np.array([0, np.pi])
    res = rho_c(th, 1, 0.1)
    assert res.shape == th.shape

def test_theta0_a_tip_length_consistency():
    b = 0.2
    base_d, tip_d = 0.06, 0.01
    th0 = theta0_from_ratio(b, base_d, tip_d)
    a = a_from_tip(b, tip_d)
    L = length_central(a, b, th0)
    assert np.isfinite(L) and L > 0

def test_L_and_f_of_b_relation():
    base_d, tip_d, L_target = 0.06, 0.01, 0.25
    b = 0.2
    Lb = L_from_b(b, base_d, tip_d)
    assert np.isclose(f_of_b(b, base_d, tip_d, L_target), Lb - L_target)

def test_factory_callable():
    params = SpiralParams(base_d=0.06, tip_d=0.01, L_target=0.25)
    f = make_f_of_b(params)
    v = f(0.2)
    assert np.isfinite(v)

def test_taper_angle_phi_range():
    phi = taper_angle_phi(0.2)
    assert 0.0 <= phi <= math.pi

def test_distance_and_footpoint():
    # Line y - 2 = 0 -> distance 2, perpendicular foot (0, 2)
    d = distance_point_line_from_origin(0.0, 1.0, -2.0)
    assert math.isclose(d, 2.0, rel_tol=1e-12)
    Mx, My = closest_point_on_line_to_origin(0.0, 1.0, -2.0)
    assert math.isclose(Mx, 0.0) and math.isclose(My, 2.0)

def test_line_unit_direction_is_unit():
    ux, uy = line_unit_direction(0.0, 1.0)  # line parallel to the x-axis
    assert math.isclose(math.hypot(ux, uy), 1.0, rel_tol=1e-12)

def test_circle_radius_for_central_angle():
    # 1 / cos(30°) = 2/√3
    r = circle_radius_for_central_angle(d=1.0, theta_deg=60.0)
    assert math.isclose(r, 2.0 / math.sqrt(3.0), rel_tol=1e-12)

def test_intersection_points_simple():
    # circle x^2+y^2=1, line y=0 -> (+-1, 0)
    C, D = intersection_points_circle_line(0.0, 1.0, 0.0, r=1.0)
    xs = sorted([C[0], D[0]])
    ys = [C[1], D[1]]
    assert math.isclose(xs[0], -1.0) and math.isclose(xs[1], 1.0)
    assert all(math.isclose(y, 0.0) for y in ys)

def test_angle_between_rays():
    assert math.isclose(angle_between_rays((1, 0), (0, 1)), 90.0, rel_tol=1e-12)

def test_solve_for_points_valid_keys_and_angle():
    res = solve_for_points(0.0, 1.0, -0.5, theta_deg=60.0)
    assert set(res.keys()) == {"d", "r", "C", "D", "winkel_deg"}
    assert 0.0 <= res["winkel_deg"] <= 180.0

def test_invalid_line_raises():
    with pytest.raises(ValueError):
        distance_point_line_from_origin(0.0, 0.0, 1.0)
