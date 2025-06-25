import numpy as np
import _math
from dataclasses import dataclass

@dataclass
class ComputationParams:
    R: float
    r: float
    L: float
    l: float
    gamma: float
    gamma_l: float = 0.0
    lam: float = 0.0
    lam_l: float = 0.0
    el: float = 0.0
    delta: float = 0.0
    a_l1: float = 0.0
    a_l2: float = 0.0
    A: float = 0.0
    B: float = 0.0
    F: float = 0.0
    theta: float = 0.0
    s_0l: float = 0.0
    psi: float = 0.0


def solve_triangle(a, b, angle):
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(angle))
    angle_a_rad = np.arcsin(a * np.sin(angle) / c)
    return c, angle_a_rad


def print_p(p: ComputationParams):
    return(f"""
𝛿 {_math.decimal_to_deg_min_sec(np.degrees(p.delta))}
Л {p.el:.3f}
𝜆ₗ {p.lam_l:.4f}
A {p.A:.3f}
B {p.B:.3f}
𝛳 {_math.decimal_to_deg_min_sec(np.degrees(p.theta))}
𝐹 {p.F:.4f}
𝛼ₗ₁ {_math.decimal_to_deg_min_sec(np.degrees(p.a_l1))}
𝛼ₗ₂ {_math.decimal_to_deg_min_sec(np.degrees(p.a_l2))}
S₀ₗ {p.s_0l:.3f}
    """)


def calculate_r(p: ComputationParams):
    mu = np.arcsin(p.R / p.l * np.sin(p.gamma))
    beta_1 = np.arcsin(p.lam * np.sin(p.gamma))
    return p.L * np.cos(beta_1) + p.l * np.cos(mu)


def S_0l(p: ComputationParams):
    p.A = p.el**2 / (p.r * p.l) * np.cos(2 * p.delta) + np.cos(2 * p.gamma)
    p.B = p.el**2 / (p.r * p.l) * np.sin(2 * p.delta) - np.sin(2 * p.gamma)
    p.theta = np.arctan(p.B / p.A)
    p.F = p.lam * (p.r / p.L) * (p.A / np.cos(p.theta))

    p.a_l1 = 0.5 * np.arctan(p.B / (p.A + p.L / (p.lam * p.r)))
    # p.a_l2 = 0.5 * np.arctan(p.B / (p.A - p.L / (p.lam * p.r))) + np.radians(180)
    b_1 = np.arcsin(p.lam * np.sin(p.gamma))
    b_l1 = np.arcsin(p.r / p.l * np.sin(b_1))

    return p.R * np.cos(p.a_l1) + p.r * np.cos(b_1) + p.l * np.cos(b_l1)


def S_pl(p: ComputationParams, alpha, S_ol, c, c_l):
    
    alpha_l = alpha
    
    term_const = S_ol - p.r * (1 - p.lam * c.k) - p.l * (1 - p.lam_l * c_l.k)
    
    master_rod_part = (c.rho2 * np.cos(2 * (alpha_l + p.gamma))
                     - c.rho4 * np.cos(4 * (alpha_l + p.gamma))
                     + c.rho6 * np.cos(6 * (alpha_l + p.gamma))
                     - c.rho8 * np.cos(8 * (alpha_l + p.gamma)))

    articulated_rod_part = (c_l.rho2 * np.cos(2 * (alpha_l - p.delta))
                          - c_l.rho4 * np.cos(4 * (alpha_l - p.delta))
                          + c_l.rho6 * np.cos(6 * (alpha_l - p.delta))
                          - c_l.rho8 * np.cos(8 * (alpha_l - p.delta)))

    return term_const - p.R * (
        np.cos(alpha_l)
        + (p.r / p.L) * master_rod_part
        + (p.el / p.L) * articulated_rod_part
    )


def compute_s(p: ComputationParams, alpha):
    p.el, p.delta = solve_triangle(p.r, p.L, p.gamma)

    p.lam = p.R / p.L
    p.lam_l = p.lam * p.el / p.l

    c = _math.compute_k_rho(p.lam)
    c_l = _math.compute_k_rho(p.lam_l)

    p.s_0l = S_0l(p)

    return S_pl(p, alpha, p.s_0l, c, c_l)


def compute_v(p: ComputationParams, alpha, omega):

    alpha_l = alpha

    c = _math.compute_k_rho(p.lam)
    c_l = _math.compute_k_rho(p.lam_l)

    master_rod_part = (2 * c.rho2 * np.sin(2 * (alpha_l + p.gamma))
                     - 4 * c.rho4 * np.sin(4 * (alpha_l + p.gamma))
                     + 6 * c.rho6 * np.sin(6 * (alpha_l + p.gamma))
                     - 8 * c.rho8 * np.sin(8 * (alpha_l + p.gamma)))

    articulated_rod_part = (2 * c_l.rho2 * np.sin(2 * (alpha_l - p.delta))
                          - 4 * c_l.rho4 * np.sin(4 * (alpha_l - p.delta))
                          + 6 * c_l.rho6 * np.sin(6 * (alpha_l - p.delta))
                          - 8 * c_l.rho8 * np.sin(8 * (alpha_l - p.delta)))

    return p.R * omega * (np.sin(alpha)
                        + (p.r / p.L) * master_rod_part
                        + (p.el / p.L) * articulated_rod_part)


def compute_j(p: ComputationParams, alpha, omega):
    alpha_l = alpha

    c = _math.compute_k_rho(p.lam)
    c_l = _math.compute_k_rho(p.lam_l)

    master_rod_part = (4 * c.rho2 * np.cos(2 * (alpha_l + p.gamma))
                     - 16 * c.rho4 * np.cos(4 * (alpha_l + p.gamma))
                     + 36 * c.rho6 * np.cos(6 * (alpha_l + p.gamma))
                     - 64 * c.rho8 * np.cos(8 * (alpha_l + p.gamma)))

    articulated_rod_part = (4 * c_l.rho2 * np.cos(2 * (alpha_l - p.delta))
                          - 16 * c_l.rho4 * np.cos(4 * (alpha_l - p.delta))
                          + 36 * c_l.rho6 * np.cos(6 * (alpha_l - p.delta))
                          - 64 * c_l.rho8 * np.cos(8 * (alpha_l - p.delta)))

    return p.R * omega**2 * (np.cos(alpha)
                        + (p.r / p.L) * master_rod_part
                        + (p.el / p.L) * articulated_rod_part)


def svj(R, L, r, l, gamma, rpm):  # path, velocity, acceleration
    omega = _math.rpm_to_rps(rpm)
    gamma = np.radians(gamma)

    params = ComputationParams(R=R, r=r, L=L, l=l, gamma=gamma)

    alpha_deg = np.linspace(0, 360, 1000)
    alpha = np.radians(alpha_deg)

    S = compute_s(params, alpha)

    V = compute_v(params, alpha, omega)
    V /= 1000  # Convert mm/s to m/s

    J = compute_j(params, alpha, omega)
    J /= 1000  # Convert mm/s2 to m/s2

    return S, V, J
