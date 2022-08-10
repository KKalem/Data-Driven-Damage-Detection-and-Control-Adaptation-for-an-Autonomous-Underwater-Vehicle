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

    # training set
    sim_realified = Dataset('normal_9_500rpm_data.json')
    sim_realified.xdot[:,:3] = realifier.sim_xformed

    # realify the validation set
    val = Dataset('normal3_9_500rpm_data.json')
    val.xdot[:,:3] = realifier.xform(val.xyz)

    # also get some real data in here
    real = Dataset('real_lolo_data.json')
    # no need to realifity the real

    # realify the extra weight dataset
    ew = Dataset('extra_weight2_9_500rpm_data.json')
    ew.xdot[:,:3] = realifier.xform(ew.xyz)

    br = Dataset('broken_angles_9_500rpm_data.json')
    br.xdot[:,:3] = realifier.xform(br.xyz)

    rud = Dataset('rudder_one_direction_9_500rpm_data.json')
    rud.xdot[:,:3] = realifier.xform(rud.xyz)
    # this dataset has the rudder (u[:,0]) at 0 to 1 but the intention was -1,1
    rud.u[ rud.u[:,0] == 0, 0] = -1
    rud.u[ np.abs(rud.u[:,0]-0.5) < 0.01, 0] = 0

    t1100 = Dataset('t1_100_9_500rpm.json')
    t1100.xdot[:,:3] = realifier.xform(t1100.xyz)

    sets = [sim_realified, val, ew, br, rud, t1100]

    # filter the idle parts out
    for d in sets:
        # d.filter(d.u[:,2] > 0.3)
        d.filter(d.u[:,3] > 0.3)

    # _, real_xdot_man4, _, _, _ = real.get_maneuver(4)
    # real_pX = real_xdot_man4[:,:3]

    maneuvers = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 0),(0, 1),(1, -1),(1, 0),(1, 1)]
    # create a one pX for all the points we are gonna predict
    # and one Y to check accuracy against
    pXs = [d.xyz for d in sets]
    Ys = [np.array([np.argmin(geom.euclid_distance((r,e), maneuvers)) for r,e,t1,t2 in d.u]) for d in sets]
    # the same thing, but this one is for "we think we are doing man X"
    # so it wont be modified to be _actually did man X_ like Ys will be
    attempted_Ys = [np.array([np.argmin(geom.euclid_distance((r,e), maneuvers)) for r,e,t1,t2 in d.u]) for d in sets]


    # the labels for the extra weight dataset should be all 9 (OOD)
    Ys[2][:] = 9
    # and the labels for the borken rudder set should be 9 too
    # beacuse both the rudder and elevator are broken
    # so the only "doable" maneuver is straight ahead
    uses_elev_or_rudder = Ys[3] != 4
    Ys[3][uses_elev_or_rudder] = 9

    # stuck rudders, all bad
    Ys[4][:] = 9
    # bad thrust, all bad
    Ys[5][:] = 9


    kdtree_detector = KDTreeDetector(sim_realified, pos_radius=0.01, ori_radius=0)
    svgp_detector = MultiSVGP() #default args are the only ones that work lol

    # sys.exit(0)


    def search_params():
        all_accs = []
        all_f1s =[]
        all_predictions = []
        for model, lax_param, harsh_param in [(svgp_detector,0.6,1.0), (kdtree_detector,0.02,0.0001)]:
            results = []
            f1s = []
            predictions = []
            for limit in np.linspace(lax_param, harsh_param, 20):
                res = [limit]
                f1 = [limit]
                for pX,Y,aY in zip(pXs, Ys, attempted_Ys):
                    pY = model.predict_maneuver(pX, limit=limit, man_id=aY)
                    correct = np.sum(pY == Y)
                    res.append(correct/len(pX))
                    f1.append(f1_score(y_true=Y, y_pred=pY, average='macro'))
                    predictions.append((limit, Y, pY))

                print(model, res)
                print(model, f1)
                results.append(res)
                f1s.append(f1)

            all_accs.append(results)
            all_predictions.append(predictions)
            all_f1s.append(f1s)


        all_accs = np.array(all_accs)
        all_predictions = np.array(all_predictions)
        all_f1s = np.array(all_f1s)

        with open('param_search_accs.npy','wb+') as f:
            np.save(f, all_accs)

        with open('param_search_predictions.npy','wb+') as f:
            np.save(f, all_predictions)

        with open('param_search_f1s.npy','wb+') as f:
            np.save(f, all_f1s)

    def confuse():
        all_predictions = []
        # limits coming from param search above
        for model, limit in [(svgp_detector, 0.89474), (kdtree_detector, 0.01267)]:
            predictions = []
            for pX,Y,aY in zip(pXs, Ys, attempted_Ys):
                pY = model.predict_maneuver(pX, limit=limit, man_id=aY)
                predictions.append((Y, pY))

            all_predictions.append(np.array(predictions))

        all_predictions = np.array(all_predictions)
        with open('param_search_predictions.npy','wb+') as f:
            np.save(f, all_predictions)


    ######
    # search_params()
    ######

    with open('param_search_accs.npy', 'rb') as f:
        all_accs = np.load(f)

    gp_accs, kd_accs = all_accs
    for results, title, xlabel in zip([gp_accs, kd_accs], ["SVGP", "KDTree"], ["Confidence", "Distance"]):
        plt.figure(figsize=(3.4/2, 3))
        plt.ylim(-0.1,1.1)
        plt.plot(results[:,0],results[:,1], label="Training")
        plt.plot(results[:,0],results[:,2], label="Validation", linestyle="--")
        plt.plot(results[:,0],results[:,3], label="Heavy", linestyle=":")
        plt.plot(results[:,0],results[:,4], label="Rud+Elev Limited", linestyle="-.")
        plt.plot(results[:,0],results[:,5], label="Rud Limited", linestyle=(0,(3,5,1,5,1,5)))
        plt.plot(results[:,0],results[:,6], label="T1 Low RPM", linestyle=(0,(5,1,5,1,1,1)))

        # plt.legend()
        # ignore training data
        acc_sum = np.sum(results[:,1:], axis=1)
        best_param_idx = np.argmax(acc_sum)
        print(f"best param = {results[best_param_idx]}")
        # plt.scatter(results[best_param_idx,0], results[best_param_idx,-1])
        plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5)
        plt.subplots_adjust(left=0.25, bottom=0.16, right=0.98, top=0.9)
        plt.title(title)
        plt.ylabel("Accuracy")
        plt.xlabel(xlabel)

    # flip the 2nd one
    plt.gca().invert_xaxis()

    sys.exit(0)

    with open('param_search_f1s.npy', 'rb') as f:
        all_f1s = np.load(f)

    gp_f1s, kd_f1s = all_f1s
    for results, title, xlabel in zip([gp_f1s, kd_f1s], ["SVGP", "KDTree"], ["Confidence", "Distance"]):
        plt.figure(figsize=(3.4/2, 3))
        plt.ylim(-0.1,1.1)
        plt.plot(results[:,0],results[:,1], label="Training")
        plt.plot(results[:,0],results[:,2], label="Validation", linestyle="--")
        plt.plot(results[:,0],results[:,3], label="Heavy", linestyle=":")
        plt.plot(results[:,0],results[:,4], label="Damaged", linestyle="-.")
        # plt.legend()
        # ignore training data
        acc_sum = np.sum(results[:,1:], axis=1)
        best_param_idx = np.argmax(acc_sum)
        print(f"best param = {results[best_param_idx]}")
        # plt.scatter(results[best_param_idx,0], results[best_param_idx,-1])
        plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5)
        plt.subplots_adjust(left=0.25, bottom=0.16, right=0.98, top=0.9)
        plt.title(title)
        plt.ylabel("F-score")
        plt.xlabel(xlabel)

    # flip the 2nd one
    plt.gca().invert_xaxis()


    confuse()
    with open('param_search_predictions.npy','rb') as f:
        all_predictions = np.load(f, allow_pickle=True)



    all_confs = []
    for model_name, model_preds in zip(['svgp', 'kdtree'], all_predictions):
        confs = []
        for i,dataset_name in enumerate(['Training', 'Validation', 'Heavy', 'Damaged']):
            ys, pys = model_preds[i]
            conf = confusion_matrix(y_true=ys, y_pred=pys, normalize='all')
            confs.append((dataset_name, conf))
            f1 = f1_score(y_true=ys, y_pred=pys, average='micro')
            print(model_name, dataset_name, f1)
        all_confs.append((model_name, confs))

    # fig, axes = plt.subplots(2,4)
    # for sets, ax_id in zip([slice(0,4), slice(4,8)], [0,1]):
        # for (title, conf), ax in zip(confs[sets], axes[ax_id]):
            # c = np.array(conf*100, dtype=int)
            # ax.imshow(c)
            # ax.set_label(title)

    for model_name, confs in all_confs:
        fig = plt.figure()
        grid = AxesGrid(fig, 111,
                        nrows_ncols=(2,2),
                        axes_pad=0.05,
                        cbar_mode='single',
                        cbar_location='right',
                        cbar_pad=0.1
                       )


        for (dataset_name, conf), ax in zip(confs, grid):
            ax.set_axis_off()
            im = ax.imshow(conf, vmin=0, vmax=0.5)

        cbar = ax.cax.colorbar(im)
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_yticks(np.arange(0, 101, 50))
        cbar.ax.set_yticklabels(['low', 'medium', 'high'])











