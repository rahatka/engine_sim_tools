import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class HarmonicCoefficients:
    lam: float
    k: float
    rho2: float
    rho4: float
    rho6: float
    rho8: float
    
@dataclass
class Harmonics:
    S1: float
    S2: float
    S4: float
    S6: float
    S8: float


def fmt(x, width=10, precision=9):
    return f"{x:.{precision}f}".rjust(width, "0")


def decimal_to_deg_min_sec(decimal_deg):
    if decimal_deg < 0:
        decimal_deg += 360
    degrees = int(decimal_deg)
    minutes_float = (decimal_deg - degrees) * 60
    minutes = int(minutes_float)
    seconds = (minutes_float - minutes) * 60
    return f"{degrees}°{minutes:02d}′{seconds:.0f}″"


def rpm_to_rps(rpm):
    return (2 * np.pi * rpm) / 60


def ms2_to_g(acc):
    return acc / 9.80665


def compute_k_rho(lam: float) -> HarmonicCoefficients:
    k =    (1/4) * lam + (3/64) * lam**3 + (5/256) * lam**5 + (175 / 128**2) * lam**7 #23
    rho2 = (1/4) * lam + (1/16) * lam**3 + (5/512) * lam**5 + (35/2048)      * lam**7 #24
    rho4 =               (1/64) * lam**3 + (3/256) * lam**5 + (35/4096)      * lam**7 #24
    rho6 =                                 (1/512) * lam**5 +  (5/2048)      * lam**7 #24
    rho8 =                                                 (5 / 128**2)      * lam**7 #24
    return HarmonicCoefficients(lam, k, rho2, rho4, rho6, rho8)


def compute_s(R, alpha, c: HarmonicCoefficients):
    return R * (1 + c.k) - R * (np.cos(alpha) #27
                                + c.rho2 * np.cos(2 * alpha)
                                - c.rho4 * np.cos(4 * alpha)
                                + c.rho6 * np.cos(6 * alpha)
                                - c.rho8 * np.cos(8 * alpha))


def compute_desaxe_s(R, L, alpha, c: HarmonicCoefficients, k):
    beta = np.arcsin(c.lam * (np.sin(alpha) - k))
    Sol = R * np.sqrt((1 / c.lam + 1)**2 - k**2)
    return Sol - (R * np.cos(alpha) + L * np.cos(beta))


def compute_v(R, alpha, omega, c: HarmonicCoefficients):
    return R * omega * (np.sin(alpha)  #32
                        + 2 * c.rho2 * np.sin(2 * alpha)
                        - 4 * c.rho4 * np.sin(4 * alpha)
                        + 6 * c.rho6 * np.sin(6 * alpha) 
                        - 8 * c.rho8 * np.sin(8 * alpha))


def compute_desaxe_v(R, alpha, omega, c: HarmonicCoefficients, k):
    beta = np.arcsin(c.lam * (np.sin(alpha) - k))
    return R * omega * (np.sin(alpha + beta) / np.cos(beta))
    

def compute_j(R, alpha, omega, c: HarmonicCoefficients):
    return R * omega**2 * (np.cos(alpha) #41
                           + 4 * c.rho2 * np.cos(2 * alpha)
                           - 16 * c.rho4 * np.cos(4 * alpha)
                           + 36 * c.rho6 * np.cos(6 * alpha)
                           - 64 * c.rho8 * np.cos(8 * alpha))


def compute_desaxe_j(R, alpha, omega, c: HarmonicCoefficients, k):
    beta = np.arcsin(c.lam * (np.sin(alpha) - k))
    return R * omega**2 * (np.cos(alpha + beta) / np.cos(beta) + c.lam * (np.cos(alpha)**2) / (np.cos(beta)**3))


def svj(R, L, rpm, offset=0):  # path, velocity, acceleration
    k = offset/R
    omega = rpm_to_rps(rpm)
    lam = R / L

    alpha_deg = np.linspace(0, 360, 1000)  # High resolution
    alpha = np.radians(alpha_deg)  # Convert to radians

    c = compute_k_rho(lam)

    S = compute_desaxe_s(R, L, alpha, c, k) if k != 0 else compute_s(R, alpha, c)

    V = compute_desaxe_v(R, alpha, omega, c, k) if k != 0 else compute_v(R, alpha, omega, c)
    V /= 1000  # Convert mm/s to m/s

    J = compute_desaxe_j(R, alpha, omega, c, k) if k != 0 else compute_j(R, alpha, omega, c)
    J /= 1000  # Convert mm/s2 to m/s2

    return S, V, J


def graphs(S, V, J):
    # color_seq = ['w', 'r', 'g', 'b', 'indianred', 'springgreen', 'cornflowerblue', 'mistyrose', 'lightgreen', 'skyblue', 'grey']
    color_seq = [
        'white',
        'red',
        'lime',
        'deepskyblue',
        'orange',
        'violet',
        'gold',
        'lightgreen',
        'lightskyblue',
        'plum',
        'grey',
        'dimgray'
    ]
    alpha_deg = np.linspace(0, 360, 1000)

    plt.style.use("dark_background")

    # Piston Position Graph
    plt.figure(figsize=(10, 5))
    for i, s in enumerate(S):
        linewidth = 0.75 if i == 0 else 0.5
        plt.plot(alpha_deg, s, label=f"Piston {i+1}", color=color_seq[i], linewidth=linewidth, zorder=-i)
    plt.xlabel("Crank Angle (°)")
    plt.ylabel("Piston Position (mm)")
    plt.title("Piston Position $S_p$")
    plt.xticks(np.arange(0, 361, 40))
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Piston Speed Graph
    plt.figure(figsize=(10, 5))
    for i, v in enumerate(V):
        linewidth = 0.75 if i == 0 else 0.5
        plt.plot(alpha_deg, v, label=f"Piston {i+1}", color=color_seq[i], linewidth=linewidth, zorder=-i)
    plt.xlabel("Crank Angle (°)")
    plt.ylabel("Piston Speed (m/s)")
    plt.title("Piston Speed $V_p$")
    plt.xticks(np.arange(0, 361, 40))
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    # Piston Acceleration Graph
    plt.figure(figsize=(10, 5))
    for i, j in enumerate(J):
        linewidth = 0.75 if i == 0 else 0.5
        plt.plot(alpha_deg, ms2_to_g(j), label=f"Piston {i+1}", color=color_seq[i], linewidth=linewidth, zorder=-i)
    plt.xlabel("Crank Angle (°)")
    plt.ylabel("Piston Acceleration (g)")
    plt.title("Piston Acceleration $J_p$")
    plt.xticks(np.arange(0, 361, 40))
    plt.legend()
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.show()
