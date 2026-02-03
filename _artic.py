import argparse
import numpy as np
import scipy.optimize as opt
import regex
import parse
import os
import json
import _math
import _math_artic1
import _math_artic2
from comment_parser import comment_parser
from tkinter.filedialog import askopenfilename

cc_to_ci = np.float64(0.0610237441)
angular_delta_window = 30
metric_delta_window = 10
json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')


class Calc:
    def __init__(self):
        self.redline = None  # Engine redline (RPM)
        self.bore = None  # Engine bore (mm)
        self.stroke = None  # Engine stroke (mm)
        self.cr = None  # Compression ratio in the master cylinder
        self.R = None  # Crank throw (mm)
        self.R1 = None  # Artic arm length (distance from artic pin to crank throw) (mm)
        self.L = None  # Master rod length (mm)
        self.L1 = None  # Artic rod length (mm)
        self.psi_deg = None  # Artic pin angle relative to the master rod axis (deg)
        self.sigma_deg = None  # Artic cylinder angle relative to the master cylinder axis (deg)
        self.chamber_vol = None  # Combustion chamber volume (mm^3)
        self.crs = []  # Compression ratios
        self.cyl_per_bank = None  # Number of cylinders per bank
        self.displacement = 0  # Total engine displacement (mm^3)
        self.comp_L1 = []  # Compensated (optimally calculated) articulating rod lengths (mm)
        self.comp_R1 = []  # Compensated (optimally calculated) articulating arm lengths (mm)
        self.comp_ign = []  # Compensated (optimally calculated) ignition events (deg)
        self.chamber_volumes = []  # Chamber volumes for all banks (the master bank is always first) (mm^3)


# neg_mod(x, mod): modulo that preserves negative remainders.
def neg_mod(x, mod):
    r = x % mod
    return r if x >= 0 or r == 0 else r - mod


# cyl_vol(bore, stroke): cylinder swept volume from bore and stroke.
def cyl_vol(bore, stroke):
    return (np.pi / 4) * bore**2 * stroke


# round_floats(o): recursively round all float values to 3 decimals.
def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def pct_diff(base, value):
    if base == 0:
        raise ValueError("base must be non-zero")
    return (value - base) / base * 100.0


def parse_cfg(fpath):
    cfg = {
        "num_of_banks": None,
        "cyl_per_bank": None,
        "psi": None,
        "sigma": None,
        "comp_R1": None,
    }

    comments = comment_parser.extract_comments(fpath, mime='text/x-c')
    metacomment = ''.join(comment.text() for comment in comments)
    matches = json_pattern.findall(metacomment)
    
    if not matches:
        raise ValueError("could not find artic_cfg")

    raw = None
    conf = None
    other_config = {}
    for s in matches:
        try:
            result = json.loads(s)
            if "artic_cfg" in result:
                raw = s
                conf = result["artic_cfg"]
                other_config = {k: v for k, v in result.items() if k != "artic_cfg"}
                break
        except json.JSONDecodeError:
            continue

    if conf is None:
        raise ValueError("Could not find 'artic_cfg' in any JSON.")

    for k in cfg.keys():
        cfg[k] = conf.get(k, cfg[k])

    return raw, cfg, other_config


# calculate_S(c, theta_deg, L1, R1, psi, sigma): piston position/displacement S for an articulated-rod cylinder.
# theta_deg is the crank angle in degrees; psi and sigma are angular offsets (radians, as used in trig calls).
# Uses the master-rod angle alpha and the articulating-rod angle beta to compute S along the cylinder axis.
def calculate_S(c, theta_deg, L1, R1, psi, sigma):
    theta = np.radians(theta_deg)
    alpha = np.arcsin((c.R * np.sin(theta)) / c.L)
    beta = np.arcsin((c.R * np.sin(sigma - theta) + R1 * np.sin(sigma - psi + alpha)) / L1)
    return c.R * np.cos(sigma - theta) + R1 * np.cos(sigma - psi + alpha) + L1 * np.cos(beta)


def find_S_extrema(c, L1, R1, psi, sigma):

    xatol = 1.0 / 3600.0  # 1 arcsec tolerance

    max_window_deg=(np.degrees(sigma) - angular_delta_window, np.degrees(sigma) + angular_delta_window)
    min_window_deg=(np.degrees(sigma) - angular_delta_window + 180, np.degrees(sigma) + angular_delta_window + 180)

    def S(theta_deg: float) -> float:
        v = calculate_S(c, theta_deg, L1, R1, psi, sigma)
        # If geometry goes invalid (arcsin domain -> nan), penalize it
        return float(v) if np.isfinite(v) else np.inf

    def negS(theta_deg: float) -> float:
        v = S(theta_deg)
        return np.inf if not np.isfinite(v) else -v

    res_max = opt.minimize_scalar(
        negS,
        bounds=max_window_deg,
        method="bounded",
        options={"xatol": xatol}
    )
    theta_max = float(res_max.x)
    S_max = float(-res_max.fun)

    res_min = opt.minimize_scalar(
        S,
        bounds=min_window_deg,
        method="bounded",
        options={"xatol": xatol}
    )
    theta_min = float(res_min.x)
    S_min = float(res_min.fun)

    return {
        "theta_max_deg": theta_max,
        "S_max": S_max,
        "theta_min_deg": theta_min,
        "S_min": S_min,
        "delta_S": S_max - S_min,
    }


# optimize_R1_for_CR(c, target_CR, psi, sigma): find the artic arm length R1 that matches a target compression ratio.
def optimize_R1_for_CR(c, target_CR, psi, sigma):
    def error_function(R1_guess):
        extrema = find_S_extrema(c, c.L1, R1_guess, psi, sigma)
        delta_S_max = extrema["S_max"] - c.L - c.R
        cyl_displacement = cyl_vol(c.bore, extrema["delta_S"])
        delta_S_max_displacement = cyl_vol(c.bore, delta_S_max)
        calculated_CR = cyl_displacement / (c.chamber_vol - delta_S_max_displacement) + 1
        return abs(calculated_CR - target_CR)

    result = opt.minimize_scalar(error_function, bounds=(c.R1 - metric_delta_window, c.R1 + metric_delta_window), method='bounded')
    return result.x


# optimize_L1_for_CR(c, target_CR, psi, sigma): find the artic rod length L1 that matches a target compression ratio.
def optimize_L1_for_CR(c, target_CR, psi, sigma):
    def error_function(L1_guess):
        extrema = find_S_extrema(c, L1_guess, c.R1, psi, sigma)
        delta_S_max = extrema["S_max"] - c.L - c.R
        cyl_displacement = cyl_vol(c.bore, extrema["delta_S"])
        delta_S_max_displacement = cyl_vol(c.bore, delta_S_max)
        calculated_CR = cyl_displacement / (c.chamber_vol - delta_S_max_displacement) + 1
        return abs(calculated_CR - target_CR)

    result = opt.minimize_scalar(error_function, bounds=(c.L1 - metric_delta_window, c.L1 + metric_delta_window), method='bounded')
    return result.x


# optimize_psi_for_CR(c, target_CR, psi, sigma): find the artic pin angle psi that matches a target compression ratio.
def optimize_psi_for_CR(c, target_CR, sigma, artic_tdc):
    def error_function(psi_guess):
        extrema = find_S_extrema(c, c.L1, c.R1, psi_guess, sigma)
        delta_S_max = extrema["S_max"] - c.L - c.R
        cyl_displacement = cyl_vol(c.bore, extrema["delta_S"])
        delta_S_max_displacement = cyl_vol(c.bore, delta_S_max)
        calculated_CR = cyl_displacement / (c.chamber_vol - delta_S_max_displacement) + 1
        return abs(calculated_CR - target_CR)

    a_min = min(sigma, np.radians(artic_tdc))
    a_max = max(sigma, np.radians(artic_tdc))
    result = opt.minimize_scalar(error_function, bounds=(a_min, a_max), method='bounded')
    return result.x


def calculate(c, psi, sigma):
    extrema = find_S_extrema(c, c.L1, c.R1, psi, sigma)

    artic_stroke = extrema["delta_S"]
    theta_at_max_S = extrema["theta_max_deg"]

    delta_S = extrema["S_max"] - c.L - c.R
    cyl_displacement = cyl_vol(c.bore, artic_stroke)
    delta_S_displacement = cyl_vol(c.bore, delta_S)
    artic_chamber_vol = c.chamber_vol - delta_S_displacement

    calc_cr = (cyl_displacement / (artic_chamber_vol) + 1)

    # if theta_at_max_S > 180:
    #     theta_at_max_S -= 360
    delta_sigma = theta_at_max_S - np.degrees(sigma)
    delta_sigma = neg_mod(delta_sigma, 360)
    
    c.displacement += cyl_displacement * c.cyl_per_bank
    artic_tdc = 90 + np.degrees(sigma) - np.degrees(np.arccos(c.R * np.sin(sigma) / c.L))
    optimal_R1 = optimize_R1_for_CR(c, c.cr, psi, sigma)
    optimal_L1 = optimize_L1_for_CR(c, c.cr, psi, sigma)
    optimal_psi = optimize_psi_for_CR(c, c.cr, sigma, artic_tdc)

    c.comp_L1.append(optimal_L1)
    c.comp_R1.append(optimal_R1)
    c.crs.append(calc_cr)
    c.comp_ign.append(delta_sigma)
    c.chamber_volumes.append(artic_chamber_vol / 1000)

    print(f"// stroke={artic_stroke:.3f} mm Δstroke={artic_stroke - c.stroke:.3f} mm")
    print(f"// TDC at {theta_at_max_S:.3f}°, Δσ={delta_sigma:.3f}°, ΔTDC={delta_S:.3f} mm, CR={calc_cr:.3f}")
    print(f"// CR-optimal params: L1={optimal_L1:.3f} or R1={optimal_R1:.3f} or ψ={np.degrees(optimal_psi):.3f}°")
    print(f"// bank {np.degrees(sigma):.3f}°, avg-optimized ψ={(np.degrees(sigma) + artic_tdc) / 2:.2f}°, artic TDC={artic_tdc:.3f}°")
    print(f"// ΔVolume={pct_diff(c.stroke, artic_stroke):.3f}%, ΔCR={pct_diff(c.cr, calc_cr):.3f}%")


def artic(fpath, graphs, common_r1):
    if fpath is None:
        fpath = askopenfilename()
    if fpath is None or fpath == '':
        return
    
    raw, cfg, other = parse_cfg(fpath)
    c = Calc()

    mr_content = ""
    with open(fpath, 'r+', encoding='utf-8') as f:
        mr_content = f.read()

        c.bore = parse.search("\nlabel bore({:f})", mr_content)[0] # type: ignore
        c.stroke = parse.search("\nlabel stroke({:f})", mr_content)[0]
        c.cr = parse.search("\nlabel compression_ratio({:f})", mr_content)[0]
        c.L = parse.search("\nlabel con_rod({:f})", mr_content)[0]
        c.L1 = parse.search("\nlabel L1({:f})", mr_content)[0]
        c.R = c.stroke / 2
        c.R1 = parse.search("\nlabel R1({:f})", mr_content)[0]
        c.redline = parse.search("\nlabel redline({:f})", mr_content)[0]
        c.psi_deg = cfg["psi"]
        c.sigma_deg = cfg["sigma"]
        c.chamber_vol = cyl_vol(c.bore, c.stroke) / (c.cr - 1)
        c.cyl_per_bank = cfg["cyl_per_bank"]
        rot = 360 / cfg["num_of_banks"]

        if c.psi_deg is None:
            c.psi_deg = [i * rot for i in range(1, cfg["num_of_banks"])]

        if c.sigma_deg is None:
            c.sigma_deg = [i * rot for i in range(1, cfg["num_of_banks"])]

        if graphs:
            S, V, J = [], [], []
            s,v,j = _math.svj(c.stroke / 2, c.L, c.redline)
            S.append(s)
            V.append(v)
            J.append(j)
            for i, psi in enumerate(c.psi_deg):
                R = c.stroke / 2
                r = c.R1 if common_r1 else cfg["comp_R1"][i]
                L = c.L
                l = c.L1
                gamma = c.sigma_deg[i]
                if psi % rot == 0:
                    s,v,j = _math_artic1.svj(R, L, r, l, gamma, c.redline)
                    S.append(s)
                    V.append(v)
                    J.append(j)
                else:
                    s,v,j = _math_artic2.svj(R, L, r, l, gamma, psi, c.redline)
                    S.append(s)
                    V.append(v)
                    J.append(j)
            _math.graphs(S, V, J, c.redline)
            return

        c.displacement = cyl_vol(c.bore, c.stroke) * c.cyl_per_bank
        c.crs.append(c.cr)
        c.chamber_volumes.append(c.chamber_vol / 1000)
        for i in range(cfg["num_of_banks"] - 1):
            print(f"\n// bank {i + 2}")
            calculate(c, np.radians(c.psi_deg[i]), np.radians(c.sigma_deg[i]))
        td = c.displacement / 1_000_000
        td_ci = c.displacement / 1000 * cc_to_ci
        print(f"\n// CR-optimal R1 avg {np.mean(c.comp_R1):.3f}")
        print(f"// CR-optimal L1 avg {np.mean(c.comp_L1):.3f}\n")
        print(f"// actual displacement: {td:.3f} L {td_ci:.2f} CI avg CR: {np.mean(c.crs):.3f}")

        cfg["comp_R1"] = c.comp_R1
        cfg["comp_ign"] = c.comp_ign
        cfg["chamber_volumes"] = c.chamber_volumes

        combined_config = {"artic_cfg": round_floats(cfg), **other}

        new_cfg = json.dumps(combined_config, indent=2, ensure_ascii=False, sort_keys=True)
        mr_content = mr_content.replace(raw, new_cfg)

        f.seek(0)
        f.truncate()
        f.write(mr_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'calc artic')
    parser.add_argument('-f', '--file', type=str, help='path to your engines (.mr)')
    parser.add_argument('-g', '--graphs', action='store_true', help='make graphs')
    parser.add_argument('-r', '--common_r1', action='store_true', help='use single R1 in graph display')
    if os.name == 'nt':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    args = parser.parse_args()
    artic(args.file, args.graphs, args.common_r1)
