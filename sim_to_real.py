#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import numpy as np
import time
import sys
import json
import pickle
from itertools import product

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #needed for '3d'

from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

# better 3d scatter
from toolbox import plotting as pltn
from toolbox import geometry as geom

from plot_data_json import Dataset, plot_xyz


def denoise_and_bias(xdot, remove_outliers=False):
    # dbscan to eliminate noise
    dbscan = DBSCAN(eps=0.003, min_samples=30, n_jobs=-1)
    dbscan.fit(xdot)
    if remove_outliers:
        # 1 is an outlier in the real data when using man 4
        # the rest of the maneuvers are shit anyways
        f = np.logical_and(dbscan.labels_ != -1, dbscan.labels_ != 1)
        denoised = xdot[f]
    else:
        denoised = xdot[dbscan.labels_ != -1]

    # and then go tru the non-noise(label>-1) samples and find a vector to each median
    # of the chunks
    chunk_vecs = []
    for label in np.unique(dbscan.labels_)[1:]:
        chunk = xdot[dbscan.labels_ == label]
        med = np.mean(chunk, axis=0)
        if all(med < 0.08):
            continue
        chunk_vecs.append(med)

    chunk_vecs = np.array(chunk_vecs)
    bias = np.mean(chunk_vecs, axis=0)

    return denoised, bias, dbscan.labels_


class Realifier:
    def __init__(self, real, sim):
        # literally every other maneuver is useless for our purposes.
        x, xdot, u, t, accels = real.get_maneuver(4)
        real_man_filtered = xdot
        # real_man_filtered = filter_maneuver(real.xdot[:,:3], real.u, 4)
        x, xdot, u, t, accels = sim.get_maneuver(4)
        sim_man_filtered = xdot
        # sim_man_filtered = filter_maneuver(sim.xdot[:,:3], sim.u, 4)

        real_denoised, real_bias, real_dbscan_labels = denoise_and_bias(real_man_filtered, remove_outliers=True)
        sim_denoised, sim_bias, sim_dbscan_labels = denoise_and_bias(sim_man_filtered)

        # scale the sim data to the same mag as real
        real_mag = np.linalg.norm(real_bias)
        sim_mag = np.linalg.norm(sim_bias)
        # and then rotate it to face the same way
        xy_angle = geom.vec2_directed_angle(sim_bias[:2], real_bias[:2]) + 0.015
        xz_angle = geom.vec2_directed_angle(sim_bias[[0,2]], real_bias[[0,2]])

        # temp rotate the real data to align with +x so we can easily find the "sideways" noise
        real_aligned = np.copy(real_denoised)
        real_aligned[:,:2] = geom.vec2_rotate(real_denoised[:,:2], -xy_angle)
        real_aligned[:,[0,2]] = geom.vec2_rotate(real_aligned[:,[0,2]], -xz_angle)

        # now we just want the noise in the yz plane
        # since the real data seems like multiple distributions, imma fit a GMM
        real_gmm = GaussianMixture(n_components = len(np.unique(real_dbscan_labels))-1)
        real_gmm.fit(real_aligned)

        # to do transformations we need... real_mag, sim_mag, real_gmm, xy_angle, xz_angle
        self.real_mag = real_mag
        self.sim_mag = sim_mag
        self.real_gmm = real_gmm
        self.xy_angle = xy_angle
        self.xz_angle = xz_angle

        self.real_denoised = real_denoised
        self.sim_xformed = self.xform(sim.xyz)

    def xform(self, xdot):
        # first, scale it to real data range
        sim_xformed = np.atleast_2d(xdot * (self.real_mag / self.sim_mag))
        # then add the yz-plane noise to it, ignore noise in x beacuse sim is good at that it seems
        # also, the noise is larger with distance from 0, so get the norms and scale with that
        sim_norms = np.linalg.norm(sim_xformed, axis=1)
        noise_weights = sim_norms/max(sim_norms)
        additive_noise = self.real_gmm.sample(n_samples=len(sim_xformed))[0][:,[1,2]] * noise_weights.reshape(-1,1)
        sim_xformed[:,[1,2]] += additive_noise
        # and then finally rotate it to the same bias of real data
        sim_xformed[:,:2] = geom.vec2_rotate(sim_xformed[:,:2], self.xy_angle)
        sim_xformed[:,[0,2]] = geom.vec2_rotate(sim_xformed[:,[0,2]], self.xz_angle)
        return sim_xformed









if __name__ == '__main__':
    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)

    normalize_u = True
    keep_underwater_only = True
    real = Dataset('real_lolo_data.json', normalize_u=normalize_u, keep_underwater_only=keep_underwater_only)
    sim = Dataset('normal_9_500rpm_data.json', normalize_u=normalize_u, keep_underwater_only=keep_underwater_only)

    # sim has so much more data, pick same number to real randomly
    # num_samples = len(real.xyz)
    # rand_inds = np.random.randint(0,sim.xdot.shape[0], min(num_samples, sim.x.shape[0]))
    # sim.filter(rand_inds)

    realifier = Realifier(real, sim)
    filename = "realifier.pickle"
    with open(filename, 'wb+') as f:
        pickle.dump(realifier, f)
        print(f"Dumped {filename}")

    real_denoised = realifier.real_denoised

    sim_xformed = realifier.xform(sim.xyz)

    single_sim_xformed = realifier.xform(sim.xyz[30])



    if 'plot' in sys.argv:
        points = [
            # real.xyz,
            real_denoised,
            # real_aligned,
            # sim.xyz,
            # sim_denoised,
            sim_xformed,
            # real_man_filtered,
            # sim_man_filtered
        ]
        p_xyz = np.vstack(points)
        p_u = None
        labels = []
        for i,p in enumerate(points):
            labels += [i]*len(p)

        win, app = plot_xyz(p_xyz, p_u, labels=labels)

        vecs = [
            # real_bias,
            # sim_bias
        ]
        for vec in vecs:
            pltn.gl_line3(win, np.array([[0,0,0], vec]))

        sys.exit(app.exec())

