# (c) oror 2023

import os
import argparse
import json
import parse
import regex
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from matplotlib.widgets import Button, Slider
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from comment_parser import comment_parser
from tkinter.filedialog import askopenfilename

ver = "0.9f"


def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def min_max_scale_array(arr: np.array) -> np.array:
    res = (arr - arr.min()) / (arr.max() - arr.min())
    return res


def mult(arr: np.array, m) -> np.array:
    return (arr * m)


def middle(arr: np.array) -> np.array:
    return (arr - 0.5)


def scale(arr: np.array, factor) -> np.array:
    return (arr * factor)


def scale_abs(arr: np.array, val: float) -> np.array:
    arr = min_max_scale_array(arr)
    return scale(arr, val)


def trim(arr: np.array, factor: float) -> np.array:
    arr = arr * (1 / factor)
    arr[arr > 1.0] = 1.0
    return arr


def setup_ax(ax):
    ax.set_theta_zero_location('N')
    ax.set_xticklabels([])
    ax.set_rlabel_position(0.0)
    ax.grid(color='dimgrey', linewidth=0.3)
    ax.tick_params(colors='darkgrey')
    ax.set_rlabel_position(180)


json_pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
def parse_cfg(fpath):
    cfg = {
        "es_version": "0.1.14a",
        "resolution": 64,
        "intake_volume": 0.5,
        "exhaust_volume": 0.5,
        "intake_at_lift": 0.0,
        "exhaust_at_lift": 0.0,
        "intake_trim": 1.0,
        "exhaust_trim": 1.0,
        "intake_sigma": 2.0,
        "exhaust_sigma": 2.0,
        "intake_base_mult": 1.66,
        "exhaust_base_mult": 1.66,
        "intake_cos": False,
        "exhaust_cos": False,
        "equal_base_radius": False,
        "ramp_steepness": 0,
        "ramp_position": 0,
        "lift_significant_fraction": 250.0,
        "roller_tappet_radius": 0.5,
    }

    comments = comment_parser.extract_comments(fpath, mime='text/x-c')
    metacomment = ''.join(comment.text() for comment in comments)
    matches = json_pattern.findall(metacomment)
    
    if not matches:
        raise ValueError("could not find cam_cfg")

    raw = None
    conf = None
    other_config = {}
    for s in matches:
        try:
            result = json.loads(s)
            if "cam_cfg" in result:
                raw = s
                conf = result["cam_cfg"]
                other_config = {k: v for k, v in result.items() if k != "cam_cfg"}
                break
        except json.JSONDecodeError:
            continue

    if conf is None:
        raise ValueError("Could not find 'cam_cfg' in any JSON.")

    for k in cfg.keys():
        cfg[k] = conf.get(k, cfg[k])

    return raw, cfg, other_config


def advertised_lift_duration(x: np.array, y: np.array) -> tuple[float, float]:
    step = x[1] - x[0]
    first_lift_index = np.argmax(y > 0)
    last_lift_index = len(y) - np.argmax(y[::-1] > 0) - 1
    l = x[first_lift_index] - step / 2
    r = x[last_lift_index] + step / 2
    return l, r


def asymmetry(y: np.array) -> np.array:
    x = np.linspace(-np.pi, np.pi, len(y)//2)
    lift_mult_map = np.cos(x)
    max = np.max(y)
    mult = 0.4
    lift_mult_map = lift_mult_map * mult
    lift_mult_map = lift_mult_map + 1 + mult
    for i, l in enumerate(y):
        if i == len(lift_mult_map):
            break
        y[i] = max if l * lift_mult_map[i] > max else l * lift_mult_map[i]
    return y


def complete_arrays(x: np.array, y: np.array) -> tuple[np.array, np.array]:
    step = x[len(x) // 2] - x[len(x) // 2 - 1]
    if not np.isclose(abs(x[0]), abs(x[-1])):
        print("x is not symmetrical")
    max_x = max(x)
    num_steps_to_180 = int(np.ceil((180 - max_x) / step))
    aug_x = max_x + step * np.arange(1, num_steps_to_180 + 1)
    rx = np.concatenate((-np.flip(aug_x), x, aug_x))
    ry = np.pad(y, (num_steps_to_180, num_steps_to_180), 'constant')
    return rx, ry


def update_lobe_function(content, x, y, base, lobe_name, header):
    def create_samples(x, y, step):
        unique_elements = np.unique(x)
        if unique_elements.size < x.size:
            print("x has duplicates")
        res = ""
        l, r = None, None
        for i, deg in enumerate(x):
            if not np.isclose(y[i], 0, atol=0.001):
                if l is None:
                    l = deg
                r = deg
                res += f"\n        .add_sample({deg:.3f} * units.deg, {y[i]:.3f} * units.mm)"
        res = f"\n        .add_sample({l - step:.3f} * units.deg, 0.000 * units.mm)" + res + f"\n        .add_sample({r + step:.3f} * units.deg, 0.000 * units.mm)"
        return res

    p_lp = regex.compile(fr'(?<=public node {lobe_name} )\{{([^{{}}]+)\}}')
    p_lip = regex.compile(fr':\s(.*?)(?=,)(?=.*{lobe_name})')
    step = x[len(x) // 2] - x[len(x) // 2 - 1]
    profile = f"""{{
    // {header}
    alias output __out: _lobe_profile;
    function _lobe_profile({step:.3f} * units.deg)
        _lobe_profile{create_samples(x,y,step)}\n}}\n"""

    base_rad = f": {base:.1f} * units.mm"
    content = p_lp.sub(profile[:-1], content)
    content = p_lip.sub(base_rad, content)
    return content


def create_profile(duration, lift, trim_factor, sigma, vol, res, at_lift, ramp_steepness, ramp_pos, lift_significant_fraction, cos):

    def restore_resolution(x: np.array, y: np.array) -> tuple[np.array, np.array]:
        min_index = np.argmax(y > 0)
        min_index = min_index if min_index == 0 else min_index - 1
        restored_space = np.linspace(x[min_index], 0, res // 2)
        f = interp1d(x[min_index:], y[min_index:], kind='cubic', fill_value='extrapolate')
        restored_values = f(restored_space)
        return restored_space, restored_values

    def create_ramp(arr: np.array, steepness: float) -> np.array:
        x = np.linspace(-steepness, steepness, res)
        s_y = min_max_scale_array(1 / (1 + np.exp(-x + ramp_pos)))
        lifts_lt_at = np.sum(arr < at_lift)
        s_x = scale_abs(x + steepness, lifts_lt_at)
        f = interp1d(s_x, s_y, kind='cubic', fill_value='extrapolate')
        indices = np.arange(lifts_lt_at)
        interpolated_values = f(indices)
        potential_lift = arr[:lifts_lt_at] * interpolated_values
        arr[:lifts_lt_at] = np.where(potential_lift < lift / lift_significant_fraction, 0.0, potential_lift)
        return arr

    x = np.linspace(-np.pi, 0, res // 2)
    vol = 1 / vol
    if cos:
        y = np.where(np.isclose(vol, 1.0), np.cos(x), np.power(vol, np.cos(x)))
        if vol < 1.0:
            y = -y
    else:
        y = np.power(vol * 10.0, -np.abs(x) / np.pi)

    x = mult(min_max_scale_array(x) - 1, duration / 4)
    y = scale_abs(gaussian_filter1d(trim(min_max_scale_array(y), trim_factor), sigma=sigma), lift)

    if at_lift != 0:
        f = interp1d(y, x, kind='cubic', assume_sorted=True)
        interpolated_x = f(at_lift)
        x = mult(x, np.min(x) / interpolated_x)
        if np.min(x) < -180:
            raise ValueError("cam lobe duration circular overlap, probably at_lift is too big")

    if ramp_steepness > 0 and at_lift > 0:
        y = create_ramp(y, ramp_steepness)
        x, y = restore_resolution(x, y)

    x = np.concatenate((x, np.flip(x[:-1]) * -1))
    y = np.concatenate((y, np.flip(y[:-1])))

    # y = asymmetry(y)
    # y = gaussian_filter1d(y, sigma=sigma)

    x, y = complete_arrays(x, y)

    return x, y


def create_valve_path_flat_tappet(xs: np.array, ys: np.array, base: float) -> np.array:
    radians_xs = np.radians(xs)
    lifts_with_base = base + ys
    ry = np.empty_like(xs)
    for i, x in enumerate(radians_xs):
        curr_cos = np.cos(radians_xs - x)
        ry[i] = np.max(curr_cos * lifts_with_base) - base
    return ry


def create_valve_path_roller_tappet(xs: np.array, ys: np.array, base: float, roller_base: float) -> np.array:
    radians_xs = np.radians(xs)
    lifts_with_base = base + ys
    ry = np.empty_like(xs)
    for i, x in enumerate(radians_xs):
        current_angles = radians_xs - x
        curr_sin = np.sin(current_angles) * lifts_with_base
        curr_cos = np.cos(current_angles) * lifts_with_base
        valid_indices = np.abs(curr_sin) <= roller_base
        b = np.sqrt(np.maximum(roller_base**2 - curr_sin[valid_indices]**2, 0))
        valid_lifts = curr_cos[valid_indices] + b
        max_lift = np.max(valid_lifts) if valid_lifts.size > 0 else 0
        ry[i] = max_lift - base - roller_base
    return ry


def plot_all(ax, fpath, cfg, int_lift, exh_lift, ivo, ivc, evo, evc):
    int_dur = 180 + ivo + ivc
    exh_dur = 180 + evo + evc
    i_c = (int_dur / 2 - ivo) / 2
    e_c = (-exh_dur / 2 + evc) / 2
    i_base = int_lift * cfg["intake_base_mult"]
    e_base = exh_lift * cfg["exhaust_base_mult"]

    if cfg["equal_base_radius"]:
        i_base = e_base = (i_base + e_base) / 2
    
    rtr = cfg["roller_tappet_radius"] * (i_base + e_base) / 2

    i_x, i_y = create_profile(int_dur, int_lift, cfg["intake_trim"], cfg["intake_sigma"], cfg["intake_volume"], cfg["resolution"],
                            cfg["intake_at_lift"], cfg["ramp_steepness"], cfg["ramp_position"], cfg["lift_significant_fraction"], cfg["intake_cos"])
    e_x, e_y = create_profile(exh_dur, exh_lift, cfg["exhaust_trim"], cfg["exhaust_sigma"], cfg["exhaust_volume"], cfg["resolution"],
                             cfg["exhaust_at_lift"], cfg["ramp_steepness"], cfg["ramp_position"], cfg["lift_significant_fraction"], cfg["exhaust_cos"])
    
    if cfg["roller_tappet_radius"] is None or cfg["roller_tappet_radius"] == 0:
        i_vly = create_valve_path_flat_tappet(i_x, i_y, i_base)
        e_vly = create_valve_path_flat_tappet(e_x, e_y, e_base)
    else:
        i_vly = create_valve_path_roller_tappet(i_x, i_y, i_base, rtr)
        e_vly = create_valve_path_roller_tappet(e_x, e_y, e_base, rtr)

    i_adv_dur = plot(ax, i_x, i_y, i_vly, i_c, i_base, int_dur, rtr, "intake", cfg["intake_at_lift"], "skyblue")
    e_adv_dur = plot(ax, e_x, e_y, e_vly, e_c, e_base, exh_dur, rtr, "exhaust", cfg["exhaust_at_lift"], "orange")

    advance_line = np.deg2rad((i_c + e_c) / 2)
    ax.plot([0, advance_line], [0, max(i_base, e_base)], linestyle='-', color='dimgrey')

    cos_formula = "f(x) = vᶜᵒˢ⁽ˣ⁾"
    exp_formula = "f(x) = vˣ"
    i_formula = cos_formula if cfg["intake_cos"] else exp_formula
    e_formula = cos_formula if cfg["exhaust_cos"] else exp_formula

    ax.text(0.05, 0.08,
             f'{os.path.basename(fpath)}\nIVO IVC: {ivo} {ivc}\nEVO EVC: {evo} {evc}'
             f'\nLSA: {i_c - e_c:.2f}\nadvance: {-e_c - i_c:.2f}\noverlap: {ivo + evc:.1f}\nbase intake formula: {i_formula}\nbase exhaust formula: {e_formula}'
             f'\nintake lift: {int_lift:.2f} mm ({int_lift * 0.03937:.3f}″)'
             f'\nexhaust lift: {exh_lift:.2f} mm ({exh_lift * 0.03937:.3f}″)'
             f'\nadvertised intake duration: {i_adv_dur:.1f}\nadvertised exhaust duration: {e_adv_dur:.1f}'
             f'\nintake duration: {int_dur:.1f} @ {cfg["intake_at_lift"]:.3f} mm ({cfg["intake_at_lift"] * 0.03937:.3f}″)'
             f'\nexhaust duration: {exh_dur:.1f} @ {cfg["exhaust_at_lift"]:.3f} mm ({cfg["exhaust_at_lift"] * 0.03937:.3f}″)'
             f'\n\nlobes.py v{ver} © oror 2023',
             transform=plt.gcf().transFigure)

    base_x = np.linspace(-np.pi, np.pi, 360)
    base_y = np.linspace(i_base, i_base, 360)
    ax.plot(base_x, base_y, linestyle='--', color='skyblue')
    base_x = np.linspace(-np.pi, np.pi, 360)
    base_y = np.linspace(e_base, e_base, 360)
    ax.plot(base_x, base_y, linestyle='--', color='orange')
    return i_x, i_vly, i_base, e_x, e_vly, e_base


def plot(ax, x, y, vly, angle, base, dur_at_lift, rtr, name, at_lift, color):
    def total_radius(arr):
        arr = np.array(arr)
        return (arr + base)
    
    def rotate(arr, angle):
        arr = np.array(arr)
        return (arr + angle)
    
    min_l, min_r = advertised_lift_duration(x, y)

    x = rotate(x, angle)
    y = total_radius(y)
    vly = total_radius(vly)

    if rtr > 0:
        num_points = 100
        theta = np.linspace(0, 2*np.pi, num_points)
        ax.plot(theta, [rtr] * num_points, color="lightgray")

    ax.plot(np.deg2rad(x), y, color=color)

    ax.plot(np.deg2rad(x), vly, linestyle='--', color=color, linewidth=0.5, label=f'{name} valve lift path')

    ax.plot([0, np.deg2rad(angle)], [0, np.max(y)], linestyle='--', color=color, linewidth=1, label=f'{name} centerline')
    ax.plot([0, np.deg2rad(min_r + angle)], [0, base], linestyle=':', color=color, linewidth=1, label=f'advertised {name} duration') #lobe start
    ax.plot([0, np.deg2rad(min_l + angle)], [0, base], linestyle=':', color=color, linewidth=1) #lobe end

    if at_lift != 0:
        ax.plot([0, np.deg2rad(dur_at_lift / 4 + angle)], [0, base + at_lift], linestyle='-', color=color, linewidth=0.5, label=f'{name} duration @ {at_lift:.3f} mm') # lobe @ lift start
        ax.plot([0, np.deg2rad(-dur_at_lift / 4 + angle)], [0, base + at_lift], linestyle='-', color=color, linewidth=0.5) # lobe @ lift end
    legend_angle = -45
    ax.legend(loc="lower left", bbox_to_anchor=(0.55 + np.cos(legend_angle) / 1.5, 0.5 + np.sin(legend_angle) / 1.5))
    return (abs(min_l) + abs(min_r)) * 2 # return advertised duration


def read_basic_params(fpath):
    mr_content = ""
    with open(fpath, 'r', encoding='utf-8') as f:
        mr_content = f.read()
    p_ivl = parse.search("\nlabel IVL({:f})", mr_content)
    p_evl = parse.search("\nlabel EVL({:f})", mr_content)
    p_ivo = parse.search("\nlabel IVO({:f} * units.deg)", mr_content)
    p_ivc = parse.search("\nlabel IVC({:f} * units.deg)", mr_content)
    p_evo = parse.search("\nlabel EVO({:f} * units.deg)", mr_content)
    p_evc = parse.search("\nlabel EVC({:f} * units.deg)", mr_content)
    return p_ivl[0], p_evl[0], p_ivo[0], p_ivc[0], p_evo[0], p_evc[0]


def lobes(fpath, arch, volume_limit, at_lift_limit):

    if fpath is None:
        fpath = askopenfilename()
    if fpath is None or fpath == '':
        return
    
    ivl, evl, ivo, ivc, evo, evc = read_basic_params(fpath)

    if arch:
        i_node_name = "Engine_Camshaft_Intake_Lobe"
        e_node_name = "Engine_Camshaft_Exhaust_Lobe"
    else:
        i_node_name = "intake_lobe_profile"
        e_node_name = "exhaust_lobe_profile"

    raw, cfg, other = parse_cfg(fpath)

    plt.style.use('dark_background')
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["font.family"] = "monospace"
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection='polar')
    setup_ax(ax)
    cf = pylab.gcf()
    cf.canvas.manager.set_window_title(os.path.basename(fpath).split('.')[0])
    i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)

    left_sliders = 0.1
    center_sliders = 0.45
    right_sliders = 0.75

    # allowed_volumes = np.concatenate([np.linspace(0.01, 0.25, 100), [0.25, 0.5, 0.75, 1.0]])
    allowed_at_lift = np.linspace(0.0, at_lift_limit, 11)
    allowed_ramp_steepness = np.linspace(0.0, 12.0, 13)
    allowed_ramp_pos = np.linspace(0.0, 6.0, 13)
    allowed_slf = [50.0, 100.0, 250.0, 500.0]
    allowed_base_radius = np.linspace(1.0, 3.0, 25)
    allowed_trim = np.linspace(0.25, 1.0, 16)
    allowed_vol = np.linspace(0.05, 1.0, 20)
    allowed_smooth = np.linspace(0.5, 5.0, 46)
    allowed_rtr = np.linspace(0.0, 2.0, 21)

    ax_i_vol = plt.axes([left_sliders, 0.96, 0.2, 0.03])
    slider_i_vol  = Slider(ax=ax_i_vol, label='volume', valmin=0.01, valmax=volume_limit, valstep=allowed_vol, valinit=cfg["intake_volume"], color='skyblue')
    ax_i_trim = plt.axes([left_sliders, 0.94, 0.2, 0.03])
    slider_i_trim = Slider(ax=ax_i_trim, label='trim', valmin=0.25, valmax=1.0, valstep=allowed_trim, valinit=cfg["intake_trim"], color='skyblue')
    ax_i_at_lift = plt.axes([left_sliders, 0.92, 0.2, 0.03])
    slider_i_at_lift = Slider(ax=ax_i_at_lift, label='@ lift mm', valmin=0.0, valmax=at_lift_limit, valstep=allowed_at_lift, valinit=cfg["intake_at_lift"], color='skyblue')
    ax_i_sigma = plt.axes([left_sliders, 0.90, 0.2, 0.03])
    slider_i_sigma = Slider(ax=ax_i_sigma, label='smoothing', valmin=0.5, valmax=5.0, valstep=allowed_smooth, valinit=cfg["intake_sigma"], color='skyblue')
    ax_i_base = plt.axes([left_sliders, 0.88, 0.2, 0.03])
    slider_i_base  = Slider(ax=ax_i_base, label='base radius', valmin=1.0, valmax=3.0, valstep=allowed_base_radius, valinit=cfg["intake_base_mult"], color='skyblue')
    ax_i_formula = fig.add_axes([left_sliders, 0.84, 0.12, 0.03])
    button_i_formula = Button(ax_i_formula, 'toggle formula', color='black', hovercolor='skyblue')

    ax_e_vol = plt.axes([right_sliders, 0.96, 0.2, 0.03])
    slider_e_vol = Slider(ax=ax_e_vol, label='volume', valmin=0.01, valmax=volume_limit, valstep=allowed_vol, valinit=cfg["exhaust_volume"], color='orange')
    ax_e_trim = plt.axes([right_sliders, 0.94, 0.2, 0.03])
    slider_e_trim = Slider(ax=ax_e_trim, label='trim', valmin=0.25, valmax=1.0, valstep=allowed_trim, valinit=cfg["exhaust_trim"], color='orange')
    ax_e_at_lift = plt.axes([right_sliders, 0.92, 0.2, 0.03])
    slider_e_at_lift = Slider(ax=ax_e_at_lift, label='@ lift mm', valmin=0.0, valmax=at_lift_limit, valstep=allowed_at_lift, valinit=cfg["exhaust_at_lift"], color='orange')
    ax_e_sigma = plt.axes([right_sliders, 0.90, 0.2, 0.03])
    slider_e_sigma = Slider(ax=ax_e_sigma, label='smoothing', valmin=0.5, valmax=5.0, valstep=allowed_smooth, valinit=cfg["exhaust_sigma"], color='orange')
    ax_e_base = plt.axes([right_sliders, 0.88, 0.2, 0.03])
    slider_e_base  = Slider(ax=ax_e_base, label='base radius', valmin=1.0, valmax=3.0, valstep=allowed_base_radius, valinit=cfg["exhaust_base_mult"], color='orange')
    ax_e_formula = fig.add_axes([right_sliders, 0.84, 0.12, 0.03])
    button_e_formula = Button(ax_e_formula, 'toggle formula', color='black', hovercolor='orange')

    ax_ramp_steepness = plt.axes([center_sliders, 0.96, 0.2, 0.03])
    slider_ramp_steepness = Slider(ax=ax_ramp_steepness, label='ramp steepness', valmin=0.0, valmax=12.0, valstep=allowed_ramp_steepness, valinit=cfg["ramp_steepness"], color='grey')
    ax_ramp_pos = plt.axes([center_sliders, 0.94, 0.2, 0.03])
    slider_ramp_pos = Slider(ax=ax_ramp_pos, label='ramp position', valmin=0.0, valmax=6.0, valstep=allowed_ramp_pos, valinit=cfg["ramp_position"], color='grey')
    ax_sign_frn = plt.axes([center_sliders, 0.92, 0.2, 0.03])
    slider_sign_frn = Slider(ax=ax_sign_frn, label='lift margin', valmin=50.0, valmax=500.0, valstep=allowed_slf, valinit=cfg["lift_significant_fraction"], color='grey')
    ax_rtr = plt.axes([center_sliders, 0.90, 0.2, 0.03])
    slider_rtr = Slider(ax=ax_rtr, label='r-tappet radius', valmin=0.0, valmax=2.0, valstep=allowed_rtr, valinit=cfg["roller_tappet_radius"], color='grey')

    ax_equal_base = fig.add_axes([0.05, 0.7, 0.14, 0.03])
    button_equal_base = Button(ax_equal_base, 'toggle equal base', color='dimgrey' if cfg["equal_base_radius"] else 'black', hovercolor='grey')

    # ax_open = fig.add_axes([0.40, 0.05, 0.07, 0.03])
    # button_open = Button(ax_open, 'open', color='black', hovercolor='grey')

    ax_save = fig.add_axes([0.435, 0.05, 0.07, 0.03])
    button_save = Button(ax_save, 'save', color='black', hovercolor='grey')

    ax_reset = fig.add_axes([0.520, 0.05, 0.07, 0.03])
    button_reset = Button(ax_reset, 'reset', color='black', hovercolor='grey')

    def reset_sliders(event):
        slider_i_vol.reset()
        slider_i_trim.reset()
        slider_i_at_lift.reset()
        slider_i_sigma.reset()
        slider_i_base.reset()
        slider_e_vol.reset()
        slider_e_trim.reset()
        slider_e_at_lift.reset()
        slider_e_sigma.reset()
        slider_e_base.reset()
        slider_ramp_steepness.reset()
        slider_ramp_pos.reset()
        slider_sign_frn.reset()
        slider_rtr.reset()

    def update_plot(val):
        nonlocal i_x, i_y, i_b, e_x, e_y, e_b
        ax.clear()
        setup_ax(ax)
        
        cfg["intake_volume"] = slider_i_vol.val
        cfg["intake_trim"] = slider_i_trim.val
        cfg["intake_at_lift"] = slider_i_at_lift.val
        cfg["intake_sigma"] = slider_i_sigma.val
        cfg["intake_base_mult"] = slider_i_base.val

        cfg["exhaust_volume"] = slider_e_vol.val
        cfg["exhaust_trim"] = slider_e_trim.val
        cfg["exhaust_at_lift"] = slider_e_at_lift.val
        cfg["exhaust_sigma"] = slider_e_sigma.val
        cfg["exhaust_base_mult"] = slider_e_base.val

        cfg["ramp_steepness"] = slider_ramp_steepness.val
        cfg["ramp_position"] = slider_ramp_pos.val
        cfg["lift_significant_fraction"] = slider_sign_frn.val
        cfg["roller_tappet_radius"] = slider_rtr.val
        try:
            i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)
        except ValueError as e:
            print(f"unable to plot the lobe: {e}")
        fig.canvas.draw_idle()

    def save(event):
        nonlocal raw, cfg, other
        combined_config = {"cam_cfg": round_floats(cfg), **other}
        header = f"lobes.py v{ver}"
        with open(fpath, 'r+', encoding='utf-8') as f:
            file_contents = f.read()
            file_contents = file_contents.replace(raw, json.dumps(combined_config, indent=2))
            file_contents = update_lobe_function(file_contents, i_x, i_y, i_b, i_node_name, header)
            file_contents = update_lobe_function(file_contents, e_x, e_y, e_b, e_node_name, header)
            f.seek(0)
            f.truncate()
            f.write(file_contents)
        raw, cfg, other = parse_cfg(fpath)

    # def reopen(event):
    #     nonlocal fpath, raw, cfg, ivl, evl, ivo, ivc, evo, evc, i_x, i_y, i_b, e_x, e_y, e_b
    #     ax.clear()
    #     setup_ax(ax)
    #     fpath = askopenfilename()
    #     ivl, evl, ivo, ivc, evo, evc = read_basic_params(fpath)
    #     raw, cfg, other = parse_cfg(fpath)
    #     reset_sliders(None)
    #     try:
    #         i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)
    #     except ValueError as e:
    #         print(f"unable to plot the lobe: {e}")
    #     fig.canvas.draw_idle()

    def toggle_i_formula(event):
        nonlocal i_x, i_y, i_b, e_x, e_y, e_b
        ax.clear()
        setup_ax(ax)
        cfg["intake_cos"] = False if cfg["intake_cos"] else True
        try:
            i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)
        except ValueError as e:
            print(f"unable to plot the lobe: {e}")
        fig.canvas.draw_idle()

    def toggle_e_formula(event):
        nonlocal i_x, i_y, i_b, e_x, e_y, e_b
        ax.clear()
        setup_ax(ax)
        cfg["exhaust_cos"] = False if cfg["exhaust_cos"] else True
        try:
            i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)
        except ValueError as e:
            print(f"unable to plot the lobe: {e}")
        fig.canvas.draw_idle()

    def toggle_equal_base(event):
        nonlocal i_x, i_y, i_b, e_x, e_y, e_b
        ax.clear()
        setup_ax(ax)
        cfg["equal_base_radius"] = False if cfg["equal_base_radius"] else True
        button_equal_base.color = 'dimgrey' if cfg["equal_base_radius"] else 'black'
        try:
            i_x, i_y, i_b, e_x, e_y, e_b = plot_all(ax, fpath, cfg, ivl, evl, ivo, ivc, evo, evc)
        except ValueError as e:
            print(f"unable to plot the lobe: {e}")
        fig.canvas.draw_idle()
    
    slider_i_vol.on_changed(update_plot)
    slider_i_trim.on_changed(update_plot)
    slider_i_at_lift.on_changed(update_plot)
    slider_i_sigma.on_changed(update_plot)
    slider_i_base.on_changed(update_plot)

    slider_e_vol.on_changed(update_plot)
    slider_e_trim.on_changed(update_plot)
    slider_e_at_lift.on_changed(update_plot)
    slider_e_sigma.on_changed(update_plot)
    slider_e_base.on_changed(update_plot)

    slider_ramp_steepness.on_changed(update_plot)
    slider_ramp_pos.on_changed(update_plot)
    slider_sign_frn.on_changed(update_plot)
    slider_rtr.on_changed(update_plot)

    button_i_formula.on_clicked(toggle_i_formula)
    button_e_formula.on_clicked(toggle_e_formula)
    button_equal_base.on_clicked(toggle_equal_base)

    # button_open.on_clicked(reopen)
    button_reset.on_clicked(reset_sliders)
    button_save.on_clicked(save)

    plt.show(block=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'calculate lobes')
    parser.add_argument('-f', '--file', type=str, help='path to the engine file (.mr)')
    parser.add_argument('-ar', '--arch', action='store_true', help='archangel motors mode')
    parser.add_argument('-vl', '--volume_limit', type=float, default=1.0, help='volume max value')
    parser.add_argument('-al', '--at_lift_limit', type=float, default=1.27, help='@ lift max value in mm')
    if os.name == 'nt':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    args = parser.parse_args()
    lobes(args.file, args.arch, args.volume_limit, args.at_lift_limit)
