import argparse
import parse
import os
import _math
from tkinter.filedialog import askopenfilename


def artic(fpath):
    if fpath is None:
        fpath = askopenfilename()
    if fpath is None or fpath == '':
        return

    mr_content = ""
    with open(fpath, 'r+', encoding='utf-8') as f:
        mr_content = f.read()

        stroke = parse.search("\nlabel stroke({:f})", mr_content)[0]
        L = parse.search("\nlabel con_rod({:f})", mr_content)[0]
        redline = parse.search("\nlabel redline({:f})", mr_content)[0]
        offset = 0
        offset_p = parse.search("\nlabel offset({:f})", mr_content)
        if offset_p is not None:
            offset = offset_p[0]

        S, V, J = [], [], []
        s,v,j = _math.svj(stroke / 2, L, redline, offset)
        S.append(s)
        V.append(v)
        J.append(j)
        s,v,j = _math.svj(stroke / 2, L, redline)
        S.append(s)
        V.append(v)
        J.append(j)
        _math.graphs(S, V, J, redline)
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'plot')
    parser.add_argument('-f', '--file', type=str, help='path to your engines (.mr)')
    if os.name == 'nt':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    args = parser.parse_args()
    artic(args.file)
