#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from kdtree_voting import KDTreeDetector
from plot_data_json import plot_xyz, Dataset
from sim_to_real import Realifier
from svgp import MultiSVGP
from plot_svgp_model import model_to_pts
from process_log import set_axes_equal

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
    nonzero = real.xyz[:,0] > 0.01
    real.filter(nonzero)

    sim = Dataset('normal_9_500rpm_data.json')
    sim.xdot[:,:3] = realifier.xform(sim.xyz)
    t1zero = sim.u[:,2] == 0
    t2zero = sim.u[:,3] == 0
    tzero = t1zero & t2zero
    sim.filter(~tzero)
    nonzero = sim.xyz[:,0] > 0.03
    sim.filter(nonzero)

    num_points = min(len(real.xyz), 500)
    real_is = np.random.randint(0,len(real.xyz), num_points)
    sim_is = np.random.randint(0,len(sim.xyz), num_points)


    fig = plt.figure(figsize=(10,10))
    plt.axis('equal')
    ax = fig.add_subplot(111, projection='3d')
    ax.azim=45
    ax.elev=30
    pltn.scatter3(ax, real.xyz[real_is], alpha=0.3, c='r')
    pltn.scatter3(ax, sim.xyz[sim_is], alpha=0.3, c='b')
    ax.set_xlabel("Forward")
    ax.set_ylabel("Sideways")
    ax.set_zlabel("Up")
