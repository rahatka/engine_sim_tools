import argparse
import numpy as np
import regex
from comment_parser import comment_parser
import json
import parse
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

def parse_cfg(fpath) -> {}:
    cfg = {
        "es_version": "0.1.14a",
        "resolution": 16,
        "start_rpm": None,
        "end_rpm": None,
        "end_deg": 0.0,
        "max_deg": None,
        "flywheel": 0,
        "exp_mode": 0,
    }

    pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
    comments = comment_parser.extract_comments(fpath, mime='text/x-c')

    metacomment = ''.join([x.text() for x in comments])
    all = pattern.findall(metacomment)
    if len(all) > 0:
        for s in all:
            try:
                result = json.loads(s)
                if result["ign_cfg"]:
                    conf = result["ign_cfg"]
                    break
            except KeyError:
                continue
        for k, _ in cfg.items():
            try:
                if conf is None:
                    raise ValueError("could not find ign_cfg")
                if conf[k] is not None:
                    cfg[k] = conf[k]
            except KeyError:
                pass
    else:
        raise ValueError("could not find ign_cfg")
    return cfg

def ignition_timing(fpath):
    if fpath is None:
        fpath = askopenfilename()
    if fpath is None or fpath == '':
        return

    cfg = parse_cfg(fpath)
    mr_content = ""
    with open(fpath, 'r', encoding='utf-8') as f:
        mr_content = f.read()
        p_redline = parse.search("\nlabel redline({:f})", mr_content)
        redline = p_redline[0]

    body_res = 64

    if cfg["end_deg"] is None:
        raise ValueError("end_deg (max advance) should be set")

    if cfg["start_rpm"] is None:
        cfg["start_rpm"] = 0
    if cfg["end_rpm"] is None:
        cfg["end_rpm"] = redline

    sr, er = cfg["start_rpm"], cfg["end_rpm"]
    fd = cfg["flywheel"]
    sd, ed, md = 0 + fd, cfg["end_deg"] + fd, cfg["max_deg"]

    if er > redline:
        slope = (ed - sd) / (er - sr)
        y_intercept = sd - slope * sr
        ed = slope * redline + y_intercept
        er = redline

    if sr > 0:
        begin_x = [0, sr]
        begin_y = [sd, sd]

    if cfg["exp_mode"] == 0:
        middle_x = [sr + 0.1, er + 0.1]
        middle_y = [sd, ed]
    elif cfg["exp_mode"] == 1:
        if sd == 0:
            sd = 1
        middle_x = np.linspace(sr, er, num=body_res)
        middle_y = np.geomspace(sd, ed, num=body_res, endpoint=True, dtype=None, axis=0)
    elif cfg["exp_mode"] == 2:
        if sd == 0:
            sd = 1
        middle_x = np.linspace(sr, er, num=body_res)
        middle_y = (sd + ed) - np.geomspace(ed, sd, num=body_res, endpoint=True, dtype=None, axis=0)
    else:
        raise ValueError("exp-mode should be in [0,1,2]")
    
    if er <= redline:
        if md is not None:
            md += fd
            end_x = [er, redline]
            end_y = [ed, md if md > ed else ed]
        else:
            end_x = [er, redline]
            end_y = [ed, ed]
    
    result_x = np.concatenate([begin_x, middle_x, end_x])
    result_y = np.concatenate([begin_y, middle_y, end_y])

    interpolated_x = np.linspace(0, redline, num=cfg["resolution"])
    interpolated_y = []
    f = interp1d(result_x, result_y, kind='linear')
    for x in interpolated_x:
        interpolated_y.append(f(x)[()])
    step = interpolated_x[1] - interpolated_x[0]
    print(f"\n    function timing_curve({step:.1f} * units.rpm)")
    print("    timing_curve")
    for i, x in enumerate(interpolated_x):
        print(f"        .add_sample({x:.1f} * units.rpm, {interpolated_y[i]:.1f} * units.deg)")
    print(f"        .add_sample({redline + step:.1f} * units.rpm, -15.0 * units.deg)")
    print(f"        .add_sample({redline + step * 2:.1f} * units.rpm, -45.0 * units.deg)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'calculate ignition')
    parser.add_argument('-f', '--file', type=str, help='path to the engine file (.mr)')

    args = parser.parse_args()
    ignition_timing(args.file)
