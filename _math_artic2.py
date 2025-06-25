import numpy as np
import matplotlib.pyplot as plt
import _math
import _math_artic1
from dataclasses import dataclass

@dataclass
class HelperParams:
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    ef: float
    theta: float


def calculate_hp(p: _math_artic1.ComputationParams):
    p.lam = p.R / p.L
    p.lam_l = p.lam * p.el / p.l
    coeff = _math.compute_k_rho(p.lam)

    A = (0.25 * p.R / p.l - 0.5 * p.lam * p.r / p.l * np.cos(p.psi) * np.cos(p.gamma) 
    + (coeff.rho2 * np.cos(p.psi) + 0.25 * p.lam * p.r / p.l * np.cos(2 * p.psi)) 
    * p.r / p.L * np.cos(2 * p.gamma))

    B = (0.5 * p.lam * p.r / p.l * np.cos(p.psi) * np.sin(p.gamma) 
        - (coeff.rho2 * np.cos(p.psi) + 0.25 * p.lam * p.r / p.l * np.cos(2 * p.psi)) 
        * p.r / p.L * np.sin(2 * p.gamma))

    C = (1 + (1 + (1 - p.lam * coeff.k - 0.5 * p.lam * coeff.rho2) * p.r / p.l * np.cos(p.psi)) 
        * p.r / p.L * np.sin(p.psi) * np.sin(p.gamma))

    D = ((1 - p.lam * coeff.k - 0.5 * p.lam * coeff.rho2) * p.r / p.l * np.sin(p.psi) 
        - (1 + (1 - p.lam * coeff.k - 0.5 * p.lam * coeff.rho2) * p.r / p.l * np.cos(p.psi)) 
        * p.r / p.L * np.sin(p.psi) * np.cos(p.gamma))

    ef = np.arctan(D / C)
    theta = np.arctan(B / A)
    E = C / np.cos(ef)
    F = 4 * A / np.cos(theta)

    return HelperParams(A, B, C, D, E, F, ef, theta)


def solve_triangle(a, b, angle):
    c = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(angle))
    angle_a_rad = np.arcsin(a * np.sin(angle) / c)
    return c, angle_a_rad


def S_0l(p: _math_artic1.ComputationParams):
    p.A = p.el**2 / (p.r * p.l) * np.cos(2 * p.delta) + np.cos(2 * p.gamma)
    p.B = p.el**2 / (p.r * p.l) * np.sin(2 * p.delta) - np.sin(2 * p.gamma)
    p.theta = np.arctan(p.B / p.A)
    p.F = p.lam * (p.r / p.L) * (p.A / np.cos(p.theta))

    p.a_l1 = 0.5 * np.arctan(p.B / (p.A + p.L / (p.lam * p.r)))
    p.a_l2 = 0.5 * np.arctan(p.B / (p.A - p.L / (p.lam * p.r))) + np.radians(180)
    b_1 = np.arcsin(p.lam * np.sin(p.gamma))
    b_l1 = np.arcsin(p.r / p.l * np.sin(b_1))

    return p.R * np.cos(p.a_l1) + p.r * np.cos(b_1) + p.l * np.cos(b_l1)


def S_pl(p: _math_artic1.ComputationParams, alpha, S_ol, beta, beta_l):
    alpha_l = alpha
    return S_ol - (p.R * np.cos(alpha_l) + p.r * np.cos(beta - p.psi) + p.l * np.cos(beta_l))


def compute_s(p: _math_artic1.ComputationParams, alpha):
    p.el, p.delta = solve_triangle(p.r, p.L, p.gamma)

    p.lam = p.R / p.L
    p.lam_l = p.lam * p.el / p.l

    coeff = _math.compute_k_rho(p.lam)

    p.s_0l = S_0l(p)

    a = -p.R/p.l * np.sin(p.gamma)
    b = p.R/p.l * (np.cos(p.gamma) - p.r/p.L * np.cos(p.psi))
    c = p.r/p.l * np.sin(p.psi)

    p_0 = 1 - 0.25 * (a**2 + b**2) - 0.5 * (1 - 0.5 * p.lam**2) * c**2
    p_1 = (1 - p.lam * coeff.k + 0.5 * p.lam * coeff.rho2) * a * c
    p_2 = 0.25 * (a**2 - b**2 + p.lam**2 * c**2)
    q_1 = (1 - p.lam * coeff.k - 0.5 * p.lam * coeff.rho2) * b * c
    q_2 = 0.5 * a * b

    A_0 = p.s_0l - (1 - p.lam * coeff.k) * p.r * np.cos(p.psi) - p.l * p_0
    A_1 = np.cos(p.gamma) - p.l / p.R * p_1
    A_2 = coeff.rho2 * p.r / p.L * np.cos(p.psi) - p.l / p.R * p_2
    B_1 = np.sin(p.gamma) + p.r / p.L * np.sin(p.psi) - p.l / p.R * q_1
    B_2 = -p.l / p.R * q_2

    alpha_comp = alpha + p.gamma

    return A_0 - p.R * (A_1 * np.cos(alpha_comp) + A_2 * np.cos(2 * alpha_comp) + B_1 * np.sin(alpha_comp)  + B_2 * np.sin(2 * alpha_comp))


def compute_s_expanded(p: _math_artic1.ComputationParams, hp: HelperParams, alpha):
    p.el, p.delta = solve_triangle(p.r, p.L, p.gamma)

    p.lam_l = p.lam * p.el / p.l

    coeff = _math.compute_k_rho(p.lam)

    p.s_0l = S_0l(p)

    a = -p.R/p.l * np.sin(p.gamma)
    b = p.R/p.l * (np.cos(p.gamma) - p.r/p.L * np.cos(p.psi))
    c = p.r/p.l * np.sin(p.psi)

    p_0 = 1 - 0.25 * (a**2 + b**2) - 0.5 * (1 - 0.5 * p.lam**2) * c**2

    A_0 = p.s_0l - (1 - p.lam * coeff.k) * p.r * np.cos(p.psi) - p.l * p_0

    alpha_l = alpha - p.gamma
    alpha_shift = alpha_l + p.gamma

    return A_0 - p.R * (hp.E * np.cos(alpha_shift + hp.ef) + 0.25 * hp.F * np.cos(2 * alpha_shift - hp.theta))


def compute_v(p: _math_artic1.ComputationParams, hp: HelperParams, alpha, omega):
    alpha_l = alpha - p.gamma
    alpha_shift = alpha_l + p.gamma

    return p.R * omega * (hp.E * np.sin(alpha_shift + hp.ef) + 0.5 * hp.F * np.sin(2 * alpha_shift - hp.theta))


def compute_j(p: _math_artic1.ComputationParams, hp: HelperParams, alpha, omega):
    alpha_l = alpha - p.gamma
    alpha_shift = alpha_l + p.gamma

    return p.R * omega**2 * (hp.E * np.cos(alpha_shift + hp.ef) + hp.F * np.cos(2 * alpha_shift - hp.theta))


def svj(R, L, r, l, gamma, gamma_l, rpm):  # path, velocity, acceleration
    omega = _math.rpm_to_rps(rpm)
    gamma = np.radians(gamma)
    gamma_l = np.radians(gamma_l)
    psi = gamma_l - gamma
    lam = R / L

    params = _math_artic1.ComputationParams(R=R, r=r, L=L, l=l, gamma=gamma, gamma_l=gamma_l, psi=psi, lam=lam)
    hp = calculate_hp(params)

    alpha_deg = np.linspace(0, 360, 1000)
    alpha = np.radians(alpha_deg)

    S = compute_s_expanded(params, hp, alpha)

    V = compute_v(params, hp, alpha, omega)
    V /= 1000  # Convert mm/s to m/s

    J = compute_j(params, hp, alpha, omega)
    J /= 1000  # Convert mm/s2 to m/s2

    return S, V, J

#     print(f"""
# A = {hp.A:.5f}
# B = {hp.B:.5f}
# C = {hp.C:.4f}
# D = {hp.D:.4f}
# E = {hp.E:.4f}
# F = {hp.D:.4f}
# Ф = {_math.decimal_to_deg_min_sec(np.degrees(hp.ef))}
# 𝛳 = {_math.decimal_to_deg_min_sec(np.degrees(hp.theta))}
#           """)
