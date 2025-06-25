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

theta_values = np.linspace(0, 360, 3601)
json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

class Calc:
    def __init__(self):
        self.bore = None
        self.stroke = None
        self.cr = None
        self.R = None
        self.R1 = None
        self.L = None
        self.L1 = None
        self.psi_deg = None
        self.sigma_deg = None
        self.chamber_vol = None
        self.crs = []
        self.cyl_per_bank = None
        self.displacement = 0
        self.comp_L1 = []
        self.comp_R1 = []
        self.comp_ign = []

def cyl_vol(bore, stroke):
    return (np.pi / 4) * bore**2 * stroke

def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o

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

cc_to_ci = np.float64(0.0610237441)

def calculate_S(c, theta_deg, L1, R1, psi, sigma):
    theta = np.radians(theta_deg)
    
    alpha = np.arcsin((c.R * np.sin(theta)) / c.L)
    beta = np.arcsin((c.R * np.sin(sigma - theta) + R1 * np.sin(sigma - psi + alpha)) / L1)
    
    return c.R * np.cos(sigma - theta) + R1 * np.cos(sigma - psi + alpha) + L1 * np.cos(beta)

def find_optimal_L1_for_S(c, psi, sigma):
    def error_function(L1_guess):
        max_S = max([calculate_S(c, theta, L1_guess, c.R1, psi, sigma) for theta in theta_values])
        return abs(max_S - (c.L + c.R))
    
    optimal_L1 = opt.minimize_scalar(error_function, bounds=(c.L1-50, c.L1+50), method='bounded')
    return optimal_L1.x

def find_optimal_R1_for_S(c, psi, sigma):
    def error_function(R1_guess):
        max_S = max([calculate_S(c, theta, c.L1, R1_guess, psi, sigma) for theta in theta_values])
        return abs(max_S - (c.L + c.R))
    
    optimal_R1 = opt.minimize_scalar(error_function, bounds=(c.R1-50, c.R1+50), method='bounded')
    return optimal_R1.x

def find_optimal_psi_for_S(c, theta_deg, sigma, artic_tdc):
    def error_function(psi_guess):
        S = calculate_S(c, theta_deg, c.L1, c.R1, np.radians(psi_guess), sigma)
        return abs(S - (c.L + c.R))

    bound_min = min(np.degrees(sigma), artic_tdc)
    bound_max = max(np.degrees(sigma), artic_tdc)
    optimal_psi = opt.minimize_scalar(error_function, bounds=(bound_min, bound_max), method='bounded')

    return optimal_psi.x

def find_optimal_R1_for_CR(c, target_CR, psi, sigma):
    def error_function(R1_guess):
        S_values = [calculate_S(c, theta, c.L1, R1_guess, psi, sigma) for theta in theta_values]
        max_S = max(S_values)
        y = max_S - min(S_values)
        delta_S = max_S - c.L - c.R
        cyl_displacement = cyl_vol(c.bore, y)
        delta_S_displacement = cyl_vol(c.bore, delta_S)
        calculated_CR = cyl_displacement / (c.chamber_vol - delta_S_displacement) + 1
        return abs(calculated_CR - target_CR)

    result = opt.minimize_scalar(error_function, bounds=(c.R1 - 10, c.R1 + 10), method='bounded')
    return result.x

def find_optimal_L1_for_CR(c, target_CR, psi, sigma):
    def error_function(L1_guess):
        S_values = [calculate_S(c, theta, L1_guess, c.R1, psi, sigma) for theta in theta_values]
        max_S = max(S_values)
        y = max_S - min(S_values)
        delta_S = max_S - c.L - c.R
        cyl_displacement = cyl_vol(c.bore, y)
        delta_S_displacement = cyl_vol(c.bore, delta_S)
        calculated_CR = cyl_displacement / (c.chamber_vol - delta_S_displacement) + 1
        return abs(calculated_CR - target_CR)

    result = opt.minimize_scalar(error_function, bounds=(c.L1 - 10, c.L1 + 10), method='bounded')
    return result.x

def calculate(c, psi, sigma):
    S_values = [calculate_S(c, theta, c.L1, c.R1, psi, sigma) for theta in theta_values]

    max_S = max(S_values)
    max_index = S_values.index(max_S)
    theta_at_max_S = theta_values[max_index]

    y = max_S - min(S_values)
    delta_S = max_S - c.L - c.R
    cyl_displacement = cyl_vol(c.bore, y)
    delta_S_displacement = cyl_vol(c.bore, delta_S)
    calc_cr = (cyl_displacement / (c.chamber_vol - delta_S_displacement) + 1)

    if theta_at_max_S > 180:
        theta_at_max_S -= 360
    delta_sigma = theta_at_max_S - np.degrees(sigma)
    print(f"y {y:.3f} mm Δy {y - c.stroke:.3f} mm")
    print(f"max S {max_S:.3f} mm at {theta_at_max_S:.1f}° Δσ {delta_sigma:.1f}° ΔS {delta_S:.3f} mm")
    
    c.displacement += cyl_displacement * c.cyl_per_bank
    artic_tdc = 90 + np.degrees(sigma) - np.degrees(np.arccos(c.R * np.sin(sigma) / c.L))
    # optimal_L1 = find_optimal_L1_for_S(c, psi, sigma)
    optimal_R1 = find_optimal_R1_for_CR(c, c.cr, psi, sigma)
    optimal_L1 = find_optimal_L1_for_CR(c, c.cr, psi, sigma)
    # optimal_psi = find_optimal_psi_for_S(c, theta_at_max_S, sigma, artic_tdc)

    c.comp_L1.append(optimal_L1)
    c.comp_R1.append(optimal_R1)
    c.crs.append(calc_cr)
    c.comp_ign.append(delta_sigma)

    print(f"CR {calc_cr:.3f} opt L1 {optimal_L1:.3f} opt R1 {optimal_R1:.3f}")
    print(f"bank {np.degrees(sigma):.3f}° avg-optimized ψ {(np.degrees(sigma) + artic_tdc) / 2:.2f}° artic TDC {artic_tdc:.3f}°")
    # print(f"optimal L1 when R1 is {R1} mm: {optimal_L1:.3f} mm")
    # print(f"optimal R1 when L1 is {L1} mm: {optimal_R1:.3f} mm")
    # print(f"optimal ψ for max_S: {optimal_psi:.1f}°\n")

def artic(fpath, graphs):
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
            rpm = 2000
            S, V, J = [], [], []
            s,v,j = _math.svj(c.stroke / 2, c.L, rpm)
            S.append(s)
            V.append(v)
            J.append(j)
            for i, psi in enumerate(c.psi_deg):
                R = c.stroke / 2
                r = cfg["comp_R1"][i]
                # r = c.R1
                L = c.L
                l = c.L1
                gamma = c.sigma_deg[i]
                if psi % rot == 0:
                    s,v,j = _math_artic1.svj(R, L, r, l, gamma, rpm)
                    S.append(s)
                    V.append(v)
                    J.append(j)
                else:
                    s,v,j = _math_artic2.svj(R, L, r, l, gamma, psi, rpm)
                    S.append(s)
                    V.append(v)
                    J.append(j)
            _math.graphs(S, V, J)
            return

        c.displacement = cyl_vol(c.bore, c.stroke) * c.cyl_per_bank
        c.crs.append(c.cr)
        for i in range(cfg["num_of_banks"] - 1):
            print(f"\ncalculating bank {i + 2}")
            calculate(c, np.radians(c.psi_deg[i]), np.radians(c.sigma_deg[i]))
        td = c.displacement / 1_000_000
        td_ci = c.displacement / 1000 * cc_to_ci
        print(f"\n// actual displacement: {td:.3f} L {td_ci:.2f} CI avg CR: {np.mean(c.crs):.3f}")
        print(f"CR-optimal R1 avg {np.mean(c.comp_R1):.3f}")
        print(f"CR-optimal L1 avg {np.mean(c.comp_L1):.3f}\n")

        cfg["comp_R1"] = c.comp_R1
        cfg["comp_ign"] = c.comp_ign

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
    if os.name == 'nt':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    args = parser.parse_args()
    artic(args.file, args.graphs)
