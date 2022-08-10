#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from kdtree_voting import KDTreeDetector
from plot_data_json import plot_xyz, Dataset
from sim_to_real import Realifier
from svgp import MultiSVGP
from plot_svgp_model import model_to_pts

from toolbox import plotting as pltn
from toolbox import geometry as geom

import sys
import rospy
import time
import pickle

import numpy as np


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from mpl_toolkits.axes_grid1 import AxesGrid
plt.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':

    np.set_printoptions(precision=5, floatmode='fixed', suppress=True)

    # prepeare the data first
    with open("realifier.pickle", 'rb+') as f:
        realifier = pickle.load(f)

    real = Dataset('real_lolo_data.json')
    # remove noise
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.003, min_samples=60, n_jobs=-1)
    clustering = dbscan.fit(real.xyz)
    labels = clustering.labels_
    print("Num labels:{}".format(len(np.unique(labels))))
    real.filter(labels >= 0)

    t1zero = real.u[:,2] == 0
    t2zero = real.u[:,3] == 0
    tzero = t1zero & t2zero
    real.filter(~tzero)


    maneuvers = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 0),(0, 1),(1, -1),(1, 0),(1, 1)]

    pX = real.xyz
    aY = np.array([np.argmin(geom.euclid_distance((r,e), maneuvers)) for r,e,t1,t2 in real.u])
    # all of these points are _supposed to be_ all "forward" since that is literally what
    # we did IRL.
    # so anything that is not 4 in this attempted Y, change to OOD for the Y
    Y = aY.copy()
    Y[aY != 4] = 9

    # training set
    sim_realified = Dataset('normal_9_500rpm_data.json')
    sim_realified.xdot[:,:3] = realifier.sim_xformed

    kdtr = KDTreeDetector(sim_realified, pos_radius=0.01, ori_radius=0)
    svgp = MultiSVGP() #default args are the only ones that work lol


    models = [svgp, kdtr]
    # limits = [0.9, 0.01]
    all_limits = [np.linspace(0.6, 1.0, 20), np.linspace(0.02, 0.0001,20)]

    all_accs = []
    all_preds = []
    for model, limits in zip(models, all_limits):
        accs = []
        preds = []
        for l in limits:
            pY = model.predict_maneuver(pX, limit=l, man_id=aY)
            acc = np.sum(pY==Y)/len(Y)
            accs.append((l, acc))
            preds.append(pY)
        all_accs.append(accs)
        all_preds.append(preds)



