#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from itertools import product
import numpy as np

# load up the model
def model_to_pts(model, n=100):
    X = model.X[model.Y == 1]
    X_colors = np.array([(1,0,0,1)]*len(X))

    Z = model.Z


    # create a volume around the training data to
    # do some predictions on
    # maxx, maxy, maxz = np.max(X, axis=0)
    # minx, miny, minz = np.min(X, axis=0)
    maxx, maxy, maxz = 0.3, 0.1, 0.05
    minx, miny, minz = 0,   0,  -0.05


    extra = 0.005
    xs = np.linspace(minx-extra, maxx+extra, n)
    ys = np.linspace(miny-extra, maxy+extra, n)
    zs = np.linspace(minz-extra, maxz+extra, n)
    prod = list(product(xs, ys, zs))
    pX = np.array(prod)


    print("Predicting...")
    pY, pYv = model.predict(pX)


    pX_colors = np.zeros(shape=(len(pX), 4))
    low = 0.95
    high = 0.99
    low_mask = pY <= low
    mid_mask = np.logical_and(pY > low, pY < high)
    high_mask = pY >= high
    pX_colors[low_mask] = [0,0,0,0]
    pX_colors[mid_mask] = [0,1,0,1]
    pX_colors[high_mask] = [0.3,0,1,1]

    # stack'em all for plotting
    pts = np.vstack([X, pX[~low_mask]])
    colors = np.vstack([X_colors, pX_colors[~low_mask]])
    return pts, colors, Z



if __name__ == '__main__':
    from plot_data_json import plot_xyz
    from toolbox import plotting as pltn
    from live_detection import make_crosshair
    from svgp import MultiSVGP

    mSVGP = MultiSVGP(name_root='svgp_model_man', num_models=9)
    pts, colors, Z = model_to_pts(mSVGP)

    print(f"About to plot {len(pts)} points")

    win, app = pltn.make_opengl_fig(grid_scale = (0.1,0.1,0.1))
    train_scatter = plot_xyz(xyz=pts, win=win, colors = colors)
    for z in Z:
        pltn.gl_line3(w=win, pts=make_crosshair(z, size=0.001, pointer=False))

    app.exec()

