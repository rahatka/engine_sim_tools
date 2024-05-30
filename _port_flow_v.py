# runner calculations are reverse engineered from
# https://www.exx.se/techinfo/runners/runners.html
# based on reverse engineered JEEP 4.0 head flow report
# https://cjclub.co.il/files/JEEP_4.0_PERFORMANCE_SPECS.pdf
# cylinder head flow figures (cfm at 28inH2O)
# all calculations are rough and approximate
# (c) oror 2024

import os
import argparse
import json
import numpy as np
import parse
import regex
from comment_parser import comment_parser
from fractions import Fraction
from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
from tkinter.filedialog import askopenfilename

ver = "1.0a"

cc_to_ci = np.float64(0.0610237441)
ci_to_cc = np.float64(16.387064069264)
si_to_cm2 = np.float64(6.451599929)
quarter_pi = np.pi / 4.0

def draw_circle(ax, center, radius, color='blue', linestyle='-'):
    circle = plt.Circle(center, radius, color=color, fill=False, linewidth=1, linestyle=linestyle)
    ax.add_artist(circle)

def setup_ax(ax):
    ax.grid(color='dimgrey', linewidth=0.3)
    ax.tick_params(colors='darkgrey')

def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o

def calculate_real_stroke(stroke, con_rod_length, offset, angle_increment=0.1):
    crank_radius = stroke / 2
    max_displacement = -float('inf')
    min_displacement = float('inf')
    angle = 0
    while angle <= 360:
        angle_rad = np.radians(angle)
        try:
            piston_position = crank_radius * np.cos(angle_rad) + np.sqrt(con_rod_length**2 - (offset + crank_radius * np.sin(angle_rad))**2)
            max_displacement = max(max_displacement, piston_position)
            min_displacement = min(min_displacement, piston_position)
        except ValueError:
            pass
        angle += angle_increment
    return max_displacement - min_displacement

def piston_position(crank_angle, crank_radius, con_rod_length, offset):
    angle_rad = np.radians(crank_angle)
    try:
        position = crank_radius * np.cos(angle_rad) + np.sqrt(con_rod_length**2 - (offset + crank_radius * np.sin(angle_rad))**2)
    except ValueError:
        position = -float('inf')
    return position

def find_tdc_angle(stroke, con_rod_length, offset):
    crank_radius = stroke / 2

    def negative_piston_position(crank_angle):
        return -piston_position(crank_angle, crank_radius, con_rod_length, offset)
    
    result = minimize_scalar(negative_piston_position, bounds=(0, 360), method='bounded')
    tdc_angle = result.x
    return 360 - tdc_angle

json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')

class Calc:
    def __init__(self):
        self.int_l = None,  # intake lifts array
        self.int_f = None,  # intake flows array
        self.exh_l = None,  # exhaust lifts array
        self.exh_f = None,  # exhaust flows array
        self.int_port_area = 0.0  # intake port area cm2
        self.int_saturated_lift = 0.0  # intake saturated lift mm
        self.exh_port_area = 0.0  # exhaust port area cm2
        self.exh_saturated_lift = 0.0  # exhaust saturated lift mm
        self.cyl_volume_cc = 0.0  # cylinder volume CC
        self.engine_volume_l = 0.0  # engine volume L
        self.harmonic = 0.0  # intake runner resonance harmonic
        self.runner_l = 0.0  # intake runner length cm
        self.runner_a = 0.0  # intake runner cross section area cm2
        self.runner_d = 0.0  # intake runner diameter cm
        self.primary_l = 0.0  # exhaust primary length cm
        self.primary_a = 0.0  # exhaust primary cross section area cm2
        self.primary_d = 0.0  # exhaust primary diameter cm
        self.collector_a = 0.0  # exhaust collector cross section area cm2
        self.collector_d = 0.0  # exhaust collector diameter cm
        self.ir_volume = 0.0  # intake runner volume cm3
        self.er_volume = 0.0  # exhaust runner volume cm3
        self.ifr = 0.0  # intake runner flow rate CFM
        self.efr = 0.0  # exhaust runner flow rate CFM
        self.i_runner_fr = 0.0  # intake runner flow rate CFM
        self.e_primary_fr = 0.0  # exhaust primary flow rate CFM
        self.plenum_vol = 0.0  # intake plenum volume L


def mm_to_inches_fraction(mm):
    inches = mm / 25.4
    whole_inches = int(inches)
    decimal_inches = inches - whole_inches
    fraction_inches = Fraction(decimal_inches).limit_denominator(64)
    if fraction_inches == 1:
        whole_inches += 1
        fraction_str = ""
    elif fraction_inches.numerator > 0:
        fraction_str = f"{fraction_inches.numerator}/{fraction_inches.denominator}"
    else:
        fraction_str = ""
    if whole_inches > 0:
        result = f"{whole_inches}" if fraction_str == "" else f"{whole_inches} {fraction_str}"
    else:
        result = fraction_str if fraction_str != "" else "0"
    return result


def parse_cfg(fpath):
    cfg = {
        "es_version": "0.1.14a",
        "resolution": 32,
        "power_factor": 1.0,
        "max_power_rpm": None,
        "max_torque_rpm": None,
        "port_to_valve_area": 0.88,
        "valve_to_stem_dia": 5.5,
        "intake_runner_dia_mult": 1.0,
        "ir_to_er_ratio": 3.0,
        "exhaust_flange_dia": None,
        "smoothness": 3.3,
    }

    comments = comment_parser.extract_comments(fpath, mime='text/x-c')
    metacomment = ''.join(comment.text() for comment in comments)
    matches = json_pattern.findall(metacomment)
    
    if not matches:
        raise ValueError("could not find flow_cfg")

    raw = None
    conf = None
    other_config = {}
    for s in matches:
        try:
            result = json.loads(s)
            if "flow_cfg" in result:
                raw = s
                conf = result["flow_cfg"]
                other_config = {k: v for k, v in result.items() if k != "flow_cfg"}
                break
        except json.JSONDecodeError:
            continue

    if conf is None:
        raise ValueError("Could not find 'flow_cfg' in any JSON.")

    for k in cfg.keys():
        cfg[k] = conf.get(k, cfg[k])

    return raw, cfg, other_config


def generate_flow(resolution, port_to_valve_ratio, head_dia, lift, saturated_l, num_valves, factor, linearity, smoothness) -> tuple[np.array, np.array]:
    lifts = np.arange(lift / resolution, lift * 2, lift / resolution)[:resolution]
    curtains = head_dia * port_to_valve_ratio * np.pi * lifts
    coeffs = 1 + np.log(saturated_l / lifts) / linearity
    coeffs[lifts > saturated_l] = 1 / (lifts[lifts > saturated_l] / saturated_l) + (lifts[lifts > saturated_l] - saturated_l) * 0.05 / np.arange(1, np.sum(lifts > saturated_l) + 0.03)
    flow_samples = curtains / factor * coeffs * num_valves
    flow_samples_with_zero = np.insert(flow_samples, 0, 0)
    pad_size = 10
    padded_flow_samples = np.pad(flow_samples_with_zero, (pad_size, 0), 'reflect', reflect_type='odd')
    smooth_flow_samples_padded = gaussian_filter1d(padded_flow_samples, sigma=smoothness)
    smooth_flow_samples = smooth_flow_samples_padded[pad_size:]
    smooth_flow_samples[0] = 0
    # lifts = lifts / 25.4
    return np.insert(lifts, 0, 0), smooth_flow_samples


def port_flow(fpath):

    if fpath is None:
        fpath = askopenfilename()
    if fpath is None or fpath == '':
        return
    
    raw, cfg, other = parse_cfg(fpath)

    c = Calc()

    mr_content = ""
    with open(fpath, 'r+', encoding='utf-8') as f:
        mr_content = f.read()

    plt.style.use('dark_background')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["font.family"] = "monospace"
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(4, 2, 4)
    ax3 = plt.subplot(2, 2, 4)
    setup_ax(ax)
    cf = pylab.gcf()
    cf.canvas.manager.set_window_title(os.path.basename(fpath).split('.')[0])

    p_ivd = parse.search("\nlabel intake_valve_diameter({:f})", mr_content)
    p_inum = parse.search("\nlabel intake_valves({:f})", mr_content)
    p_evd = parse.search("\nlabel exhaust_valve_diameter({:f})", mr_content)

    p_enum = parse.search("\nlabel exhaust_valves({:f})", mr_content)

    p_bore = parse.search("\nlabel bore({:f})", mr_content)
    p_stroke = parse.search("\nlabel stroke({:f})", mr_content)
    p_cr = parse.search("\nlabel compression_ratio({:f})", mr_content)
    p_cyl = parse.search("\nlabel cyl({:f})", mr_content)
    p_redline = parse.search("\nlabel redline({:f})", mr_content)

    p_ivl = parse.search("\nlabel IVL({:f})", mr_content)
    p_evl = parse.search("\nlabel EVL({:f})", mr_content)
    p_ivo = parse.search("\nlabel IVO({:f} * units.deg)", mr_content)
    p_ivc = parse.search("\nlabel IVC({:f} * units.deg)", mr_content)
    p_evo = parse.search("\nlabel EVO({:f} * units.deg)", mr_content)
    p_oft = parse.search("\nlabel offset({:f})", mr_content)
    p_crl = parse.search("\nlabel con_rod({:f})", mr_content)

    int_num = p_inum[0]
    int_head_dia = p_ivd[0]
    int_lift = p_ivl[0]

    exh_num = p_enum[0]
    exh_head_dia = p_evd[0]
    exh_lift = p_evl[0]

    bore = p_bore[0]
    stroke = p_stroke[0]
    cr = p_cr[0]
    cyl = p_cyl[0]
    redline = p_redline[0]
    ivo = p_ivo[0]
    ivc = p_ivc[0]
    evo = p_evo[0]
    crl = p_crl[0]

    offset = 0
    if p_oft is not None:
        offset = p_oft[0]

    if offset != 0:
        real_stroke = calculate_real_stroke(stroke, crl, offset)
        print(f"TDC comp: {find_tdc_angle(stroke, crl, offset):.3f}")
    else:
        real_stroke = stroke

    resolution = cfg["resolution"] - 1
    if cfg["max_power_rpm"] is None:
        cfg["max_power_rpm"] = redline
    if cfg["max_torque_rpm"] is None:
        cfg["max_torque_rpm"] = redline * 0.66

    allowed_pf = np.linspace(0.0, 2.0, 41)
    allowed_pva = np.linspace(0.6, 1.0, 41)
    allowed_vsd = np.linspace(3.0, 7.0, 41)
    allowed_irdm = np.linspace(0.8, 1.2, 21)
    allowed_irerr = np.linspace(1.0, 5.0, 41)
    allowed_smooth = np.linspace(1.0, 5.0, 41)

    left_sliders = 0.75
    ax_i_pf = plt.axes([left_sliders, 0.92, 0.2, 0.03])
    slider_i_pf  = Slider(ax=ax_i_pf, label='flow rate mult', valmin=0.0, valmax=2.0, valstep=allowed_pf, valinit=cfg["power_factor"], color='skyblue')
    ax_i_pva = plt.axes([left_sliders, 0.90, 0.2, 0.03])
    slider_i_pva = Slider(ax=ax_i_pva, label='port to valve area', valmin=0.6, valmax=1.0, valstep=allowed_pva, valinit=cfg["port_to_valve_area"], color='skyblue')
    ax_i_vsd = plt.axes([left_sliders, 0.88, 0.2, 0.03])
    slider_i_vsd  = Slider(ax=ax_i_vsd, label='valve to stem dia', valmin=3.0, valmax=7.0, valstep=allowed_vsd, valinit=cfg["valve_to_stem_dia"], color='skyblue')
    ax_i_irdm = plt.axes([left_sliders, 0.86, 0.2, 0.03])
    slider_i_irdm  = Slider(ax=ax_i_irdm, label='intake runner dia mult', valmin=0.8, valmax=1.2, valstep=allowed_irdm, valinit=cfg["intake_runner_dia_mult"], color='skyblue')
    ax_i_irerr = plt.axes([left_sliders, 0.84, 0.2, 0.03])
    slider_i_irerr  = Slider(ax=ax_i_irerr, label='intake to exhaust runner ratio', valmin=1.0, valmax=5.0, valstep=allowed_irerr, valinit=cfg["ir_to_er_ratio"], color='skyblue')
    ax_i_smooth = plt.axes([left_sliders, 0.82, 0.2, 0.03])
    slider_i_smooth  = Slider(ax=ax_i_smooth, label='smoothness', valmin=1.0, valmax=5.0, valstep=allowed_smooth, valinit=cfg["smoothness"], color='skyblue')

    ax_save = fig.add_axes([left_sliders, 0.78, 0.12, 0.03])
    button_save = Button(ax_save, 'save', color='black', hovercolor='skyblue')

    def plot_all():
        nonlocal c
        ax.plot(c.int_l, c.int_f, color='skyblue')
        ax.plot(c.exh_l, c.exh_f, color='orange')
        ax.vlines(x=max(c.int_l), ymin=0, ymax=max(c.int_f), linestyle='--', colors='skyblue', linewidth=0.5)
        ax.hlines(y=max(c.int_f), xmin=0, xmax=max(c.int_l), linestyle='--', colors='skyblue', linewidth=0.5)
        ax.vlines(x=max(c.exh_l), ymin=0, ymax=max(c.exh_f), linestyle='--', colors='orange', linewidth=0.5)
        ax.hlines(y=max(c.exh_f), xmin=0, xmax=max(c.exh_l), linestyle='--', colors='orange', linewidth=0.5)

        ax2.axis([0, 10, 0, 8])
        ax2.text(0.2, 7.0, f'bore x stroke {bore} x {stroke}', fontsize=8)
        ax2.text(9.8, 7.0, f'{mm_to_inches_fraction(bore)} x {mm_to_inches_fraction(stroke)}', fontsize=8, horizontalalignment='right')
        ax2.text(0.2, 6.0, f'intake valve  {int_head_dia} : {int_lift}', fontsize=8)
        ax2.text(9.8, 6.0, f'{mm_to_inches_fraction(int_head_dia)} : {mm_to_inches_fraction(int_lift)}', fontsize=8, horizontalalignment='right')
        ax2.text(0.2, 5.0, f'exhaust valve {exh_head_dia} : {exh_lift}', fontsize=8)
        ax2.text(9.8, 5.0, f'{mm_to_inches_fraction(exh_head_dia)} : {mm_to_inches_fraction(exh_lift)}', fontsize=8, horizontalalignment='right')
        ax2.text(0.2, 4.0, f'flow rates    {c.ifr:.2f} {c.efr:.2f}', fontsize=8)
        ax2.text(9.8, 4.0, f'ir_fr / ep_fr {c.i_runner_fr:.2f} {c.e_primary_fr:.2f}', fontsize=8, horizontalalignment='right')
        ax2.text(0.2, 3.0, f'saturated l   {c.int_saturated_lift:.2f} {c.exh_saturated_lift:.2f}', fontsize=8)
        ax2.text(0.2, 2.0, f'ir_v / er_v   {c.ir_volume:.2f} {c.er_volume:.2f}', fontsize=8)
        ax2.text(9.8, 2.0, f'ir_l / er_l   {c.runner_l:.2f} {c.primary_l:.2f}', fontsize=8, horizontalalignment='right')
        ax2.text(0.2, 1.0, f'volume        {c.engine_volume_l:.2f}', fontsize=8)
        ax2.text(9.8, 1.0, f'{c.engine_volume_l * 1000 / ci_to_cc:.1f}', fontsize=8, horizontalalignment='right')

        ax3.axis([0, 10, 0, 10])
        bore_radius = bore / 2
        intake_radius = int_head_dia / 2
        exhaust_radius = exh_head_dia / 2
        int_stem_radius = intake_radius / cfg["valve_to_stem_dia"]
        exh_stem_radius = exhaust_radius / cfg["valve_to_stem_dia"]
        int_port_radius = np.sqrt(((np.pi * intake_radius**2) * cfg["port_to_valve_area"]) / np.pi)
        exh_port_radius = np.sqrt(((np.pi * exhaust_radius**2) * cfg["port_to_valve_area"]) / np.pi)
        ax3.set_aspect('equal')
        draw_circle(ax3, (0, 0), bore_radius, 'white')
        total_valve_width = intake_radius + exhaust_radius
        center_distance = (bore_radius - total_valve_width) / 2 + total_valve_width
        draw_circle(ax3, (-center_distance + intake_radius, 0), intake_radius, 'skyblue')
        draw_circle(ax3, (-center_distance + intake_radius, 0), int_stem_radius, 'skyblue')
        draw_circle(ax3, (-center_distance + intake_radius, 0), int_port_radius, 'skyblue', linestyle='--')
        draw_circle(ax3, (center_distance - exhaust_radius, 0), exhaust_radius, 'orange')
        draw_circle(ax3, (center_distance - exhaust_radius, 0), exh_stem_radius, 'orange')
        draw_circle(ax3, (center_distance - exhaust_radius, 0), exh_port_radius, 'orange', linestyle='--')
        ax3.set_xlim(-bore_radius - 1, bore_radius + 1)
        ax3.set_ylim(-bore_radius - 1, bore_radius + 1)


    def update_calc():
        nonlocal c

        ipf = (1 + cfg["power_factor"]) / 2
        epf = (3 + cfg["power_factor"]) / 4

        port_to_valve_head_radius = np.sqrt(cfg["port_to_valve_area"])

        int_stem_area = quarter_pi * (int_head_dia / cfg["valve_to_stem_dia"]) ** 2
        c.int_port_area = quarter_pi * int_head_dia ** 2 * cfg["port_to_valve_area"] - int_stem_area
        c.int_saturated_lift = c.int_port_area / (int_head_dia * port_to_valve_head_radius * np.pi)

        exh_stem_area = quarter_pi * (exh_head_dia / cfg["valve_to_stem_dia"]) ** 2
        c.exh_port_area = quarter_pi * exh_head_dia ** 2 * cfg["port_to_valve_area"] - exh_stem_area
        c.exh_saturated_lift = c.exh_port_area / (exh_head_dia * port_to_valve_head_radius * np.pi)

        c.harmonic = 8
        intake_duration = 180 + ivo + ivc
        c.cyl_volume_cc = quarter_pi * bore ** 2 * real_stroke / 1000
        r2 = c.cyl_volume_cc * cfg["max_power_rpm"] / np.pi / 2.54 / 88200
        c.runner_d = np.sqrt(r2) * 2 * cfg["intake_runner_dia_mult"]
        c.runner_l = ((720 - intake_duration) * 360 / (12 * cfg["max_power_rpm"]) + 0.003 * c.runner_d) / c.harmonic * 100
        c.runner_a = quarter_pi * c.runner_d ** 2

        while c.runner_l > 20:
            if c.harmonic == 32:
                break
            c.harmonic *= 2
            c.runner_l /= 2

        c.ir_volume = c.runner_a * c.runner_l
        c.er_volume = c.ir_volume / cfg["ir_to_er_ratio"]
        arbitrary_primary_coeff = 1 / 3 * 5
        c.primary_l = ((850 * (360 - evo) / cfg["max_power_rpm"]) - 3) * 2.54 * 0.5
        c.primary_a = c.cyl_volume_cc * cc_to_ci * (cfg["max_torque_rpm"] * arbitrary_primary_coeff) / 88200 * si_to_cm2
        c.primary_d =  np.sqrt(c.primary_a / np.pi) * 2
        if cfg["exhaust_flange_dia"] is None:
            c.collector_a = c.primary_a * cyl * (2/3)
            c.collector_d =  np.sqrt(c.collector_a / np.pi) * 2
        else:
            c.collector_d = cfg["exhaust_flange_dia"] / 10
            c.collector_a = np.pi * pow(c.collector_d / 2, 2)
        c.engine_volume_l = c.cyl_volume_cc / 1000 * cyl
        c.ifr = c.engine_volume_l * redline / 60  * ipf
        c.efr = c.ifr / 3 * 5 * epf
        c.i_runner_fr = c.ifr / (cyl / 2) * (ipf if ipf > 1 else 1)
        c.e_primary_fr = c.efr / (cyl / 2) * (epf if epf > 1 else 1)
        c.plenum_vol = c.engine_volume_l * cfg["max_power_rpm"] / 6666.666 * ((1 + cfg['power_factor']) / 2)
        c.int_l, c.int_f = generate_flow(resolution, port_to_valve_head_radius, int_head_dia, int_lift, c.int_saturated_lift, int_num, 7, 3, cfg['smoothness'])
        c.exh_l, c.exh_f = generate_flow(resolution, port_to_valve_head_radius, exh_head_dia, exh_lift, c.exh_saturated_lift, exh_num, 6, 3, cfg['smoothness'])

    def update_plot(val):
        nonlocal c
        ax.clear()
        ax2.clear()
        ax3.clear()
        setup_ax(ax)
        
        cfg["power_factor"] = slider_i_pf.val
        cfg["port_to_valve_area"] = slider_i_pva.val
        cfg["valve_to_stem_dia"] = slider_i_vsd.val
        cfg["intake_runner_dia_mult"] = slider_i_irdm.val
        cfg["ir_to_er_ratio"] = slider_i_irerr.val
        cfg['smoothness'] = slider_i_smooth.val

        try:
            update_calc()
            plot_all()
        except ValueError as e:
            print(f"unable to plot: {e}")
        fig.canvas.draw_idle()

    def print_flow(lifts, flow_samples):
        res = ""
        res += "\n        ".join([f".add_flow_sample({lift:.3f}, {sample:.1f})" for lift, sample in zip(lifts, flow_samples)])
        return res + "\n"
    
    def save(val):
        nonlocal c, raw, cfg, other
        head_flow = f"""{{
    // port_flow.py v{ver}
    // intake port area: {c.int_port_area * int_num / 100:.1f} cm²; saturated lift: {c.int_saturated_lift:.2f} mm
    // exhaust port area: {c.exh_port_area * exh_num / 100:.1f} cm²; saturated lift: {c.exh_saturated_lift:.2f} mm
    // cylinder volume: {c.cyl_volume_cc:.1f} cm³ ({c.cyl_volume_cc / ci_to_cc:.1f} CI); engine volume: {c.engine_volume_l:.3f} L ({c.engine_volume_l * 1000 / ci_to_cc:.1f} CI)
    // {c.harmonic} harmonic intake runner length: {c.runner_l:.1f} cm; diameter: {c.runner_d:.1f} cm
    // primary length: {c.primary_l:.1f} cm, area: {c.primary_a:.1f} cm², diameter: {c.primary_d:.1f} cm
    // collector diameter: {c.collector_d:.1f} cm, area: {c.collector_a:.1f} cm²
    // target power: {cfg["max_power_rpm"]:.0f} RPM, power factor {cfg["power_factor"]:.2f}

    input intake_camshaft;
    input exhaust_camshaft;
    input flip_display: false;
    
    alias output __out: head;
    function intake_flow({int_lift / resolution:.3f} * units.mm)
    intake_flow
        {print_flow(c.int_l, c.int_f)}
    function exhaust_flow({exh_lift / resolution:.3f} * units.mm)
    exhaust_flow
        {print_flow(c.exh_l, c.exh_f)}
    generic_cylinder_head head(
        chamber_volume: {c.cyl_volume_cc / cr:.3f} * units.cc,
        intake_runner_volume: {c.ir_volume:.1f} * units.cc,
        intake_runner_cross_section_area: {c.runner_a:.1f} * units.cm2,
        exhaust_runner_volume: {c.er_volume:.1f} * units.cc,
        exhaust_runner_cross_section_area: {c.primary_a:.1f} * units.cm2,

        intake_port_flow: intake_flow,
        exhaust_port_flow: exhaust_flow,
        valvetrain: standard_valvetrain(
            intake_camshaft: intake_camshaft,
            exhaust_camshaft: exhaust_camshaft
        ),
        flip_display: flip_display
    )\n}}"""

        head_pattern = r"(?<=public node head )\{([^{}]+)\}"
        p_hp = regex.compile(head_pattern)
        matches = p_hp.findall(mr_content)
        if len(matches) > 0:
            with open(fpath, 'r+', encoding='utf-8') as f:
                file_contents = f.read()
                p_ptl = regex.compile(r"primary_tube_length: (.*?)(?=\n|$)")
                p_pv = regex.compile(r"plenum_volume: (.*?)(?=\n|$)")
                p_csa = regex.compile(r"plenum_cross_section_area: (.*?)(?=\n|$)")
                p_rl = regex.compile(r"runner_length: (.*?)(?=\n|$)")
                p_ifr = regex.compile(r"intake_flow_rate: (.*?)(?=\n|$)")
                p_irfr = regex.compile(r"runner_flow_rate: (.*?)(?=\n|$)")
                p_ft = regex.compile(r"friction_torque: (.*?)(?=\n|$)")

                file_contents = p_hp.sub(head_flow, file_contents)
                file_contents = p_ptl.sub(f"primary_tube_length: {c.primary_l:.1f} * units.cm,", file_contents)
                file_contents = p_pv.sub(f"plenum_volume: {c.plenum_vol:.1f} * units.L,", file_contents)
                file_contents = p_csa.sub(f"plenum_cross_section_area: {c.runner_a * cyl / 2:.1f} * units.cm2,", file_contents)
                file_contents = p_rl.sub(f"runner_length: {c.runner_l:.1f} * units.cm,", file_contents)
                file_contents = p_ifr.sub(f"intake_flow_rate: k_carb({c.ifr:.1f}),", file_contents)
                file_contents = p_irfr.sub(f"runner_flow_rate: k_carb({c.i_runner_fr:.1f}),", file_contents)

                num_of_cranks = len(regex.findall(r"(?<!//\s)\n\s*crankshaft ", file_contents))
                file_contents = p_ft.sub(f"friction_torque: {c.engine_volume_l * 1.5 / num_of_cranks:.1f} * units.Nm,", file_contents) # rough relation

                p_es_blocks = regex.compile(r"\{[^{}]* exhaust_system [^{}]*\}")
                es_blocks = p_es_blocks.findall(file_contents, regex.DOTALL)

                es_replacements = []

                for block in es_blocks:
                    num_of_exh_systems = len(regex.findall(r"(?<!//\s)\n\s*exhaust_system ", block))
                    p_ea_l = regex.compile(r"label collector_cross_section_area(.*?)(?=\n|$)")
                    p_ea = regex.compile(r"collector_cross_section_area: (.*?)(?=\n|$)")
                    p_efr = regex.compile(r"outlet_flow_rate: (.*?)(?=\n|$)")
                    p_erfr = regex.compile(r"primary_flow_rate: (.*?)(?=\n|$)")
                    replacement = block
                    replacement = p_ea_l.sub(f"label collector_cross_section_area({c.collector_a / num_of_exh_systems:.1f})", replacement)
                    replacement = p_ea.sub(f"collector_cross_section_area: {c.collector_a / num_of_exh_systems:.1f} * units.cm2,", replacement)
                    replacement = p_efr.sub(f"outlet_flow_rate: k_carb({c.efr / num_of_exh_systems:.1f}),", replacement)
                    replacement = p_erfr.sub(f"primary_flow_rate: k_carb({c.e_primary_fr:.1f}),", replacement)
                    es_replacements.append(replacement)
                
                for i, block in enumerate(es_blocks):
                    file_contents = file_contents.replace(block, es_replacements[i])

                combined_config = {"flow_cfg": round_floats(cfg), **other}
                file_contents = file_contents.replace(raw, json.dumps(combined_config, indent=2))

                f.seek(0)
                f.truncate()
                f.write(file_contents)

                raw, cfg, other = parse_cfg(fpath)
        else:
            print("no changes made")

    slider_i_pf.on_changed(update_plot)
    slider_i_pva.on_changed(update_plot)
    slider_i_vsd.on_changed(update_plot)
    slider_i_irdm.on_changed(update_plot)
    slider_i_irerr.on_changed(update_plot)
    slider_i_smooth.on_changed(update_plot)
    button_save.on_clicked(save)
    update_calc()
    plot_all()
    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'calc flow')
    parser.add_argument('-f', '--file', type=str, help='path to your engines (.mr)')
    if os.name == 'nt':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    args = parser.parse_args()
    port_flow(args.file)