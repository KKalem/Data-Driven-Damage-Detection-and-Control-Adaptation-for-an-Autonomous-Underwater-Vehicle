#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from plot_data_json import plot_xyz, Dataset

import numpy as np
import time
import sys
import json

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #needed for '3d'

from scipy.spatial import KDTree
from tqdm import tqdm

# better 3d scatter
from toolbox import plotting as pltn
from toolbox import geometry as geom


class KDTreeDetector:
    def __init__(self,
                 dataset,
                 pos_radius,
                 ori_radius):
        """
        Just an object to tidy up stuff
        """

        self.normal = dataset
        t0 = time.time()
        print("Building KDTrees")
        self.pos_kdt = KDTree(data = self.normal.xdot[:,:3])
        self.ori_kdt = KDTree(data = self.normal.xdot[:,3:])
        print("KDT build time:{:.2f}".format(time.time()-t0))

        self.pos_radius = pos_radius
        self.ori_radius = ori_radius


    def query(self, query_xdots, query_u):
        query_xdots = np.atleast_2d(query_xdot)

        pos_neighbor_indices = []
        ori_neighbor_indices = []
        for query_xdot in query_xdots:
            pos_neighbor_indices += self.pos_kdt.query_ball_point(x = query_xdot[:3], r = self.pos_radius)
            ori_neighbor_indices += self.ori_kdt.query_ball_point(x = query_xdot[3:], r = self.ori_radius)

        # now that we got the neighbors for position and orientation separately, 
        # we can merge them, ask them their controls
        # then compare those controls to the query_xdot controls
        # neighbor_indices = np.unique(pos_neighbor_indices + ori_neighbor_indices)
        neighbor_indices = pos_neighbor_indices + ori_neighbor_indices

        if len(neighbor_indices) == 0:
            return -1, None, []

        neighbor_dists = np.linalg.norm(self.normal.xdot[neighbor_indices] - query_xdot, axis=1)
        vote_weigths = 1/neighbor_dists
        normal_neighbor_us = self.normal.u[neighbor_indices]

        # imma assume there are a finite number of us in here.
        u_votes = {}
        for u, vote_w in zip(normal_neighbor_us, vote_weigths):
            u = tuple(u)
            if u not in u_votes:
                u_votes[u] = vote_w
            else:
                u_votes[u] += vote_w

        # make the votes into an array for easy sorting etc.
        u_votes_arr = []
        for u,votes in u_votes.items():
            u_votes_arr.append(list(u) + [votes])
        u_votes_arr = np.array(u_votes_arr)

        # and finally pick the most voted one
        most_voted_u = u_votes_arr[np.argmax(u_votes_arr[:,-1])][:-1]

        # and see if these match
        # print("Query u={}, voted u={}".format(query_u, most_voted_u))
        dist = geom.euclid_distance(most_voted_u, query_u)
        # print("Diff:{}".format(dist))
        return dist, most_voted_u, neighbor_indices



    def predict_maneuver(self, pX, limit=0.02, man_id=None):
        pX = np.atleast_2d(pX)
        # first acquire the neighbors
        k = 20
        dists, inds = self.pos_kdt.query(pX[:,:3], k=k, distance_upper_bound=limit)
        # weight them by distance
        ws = 1/(dists+0.00001)

        # these should be the same as get_maneuver of dataset
        maneuvers = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 0),(0, 1),(1, -1),(1, 0),(1, 1)]

        # only let the points that have neighbors into the voting
        # this is how query indicates "no neighbor"
        out_pX_inds = np.any(inds >= len(self.normal.u), axis=1)

        # these are the points that will be voted for
        inds = inds[~out_pX_inds]

        # this is (N,k,2)
        us = self.normal.u[inds][...,:2]

        # if all the data is garbage, early return
        if len(us) == 0:
            print("kdtree has no unfiltered us")
            return np.array([9]*len(pX))


        # flatten, so we can distance all at once
        flattened_us = np.reshape(us, newshape=(us.shape[0]*us.shape[1], 2))
        # distance
        man_dists = np.array([geom.euclid_distance(flattened_us,man) for man in maneuvers])
        # unflatten
        man_dists = np.reshape(man_dists, newshape = (9, us.shape[0], us.shape[1]))
        # now we have (num_man, N, k) distances here
        # now to find the argmin of the num_man parts
        man_votes = np.argmin(man_dists, axis=0)
        # (N,k) maneuver ids acqured~

        # expand the votes into the full shape now
        expanded_man_votes = np.zeros(shape=(len(pX), k), dtype=int)
        # and fill it in from the votes cast
        expanded_man_votes[~out_pX_inds] = man_votes
        # fill the out of bounds ones with 9s
        expanded_man_votes[out_pX_inds] = 9

        # create this (N,k,10) thing to put the man_votes into bit masks
        mask_votes = np.zeros(shape=(len(pX), k, len(maneuvers)+1))

        # and then this does magic.
        # it takes the indices from expanded_man_votes,
        # and puts the values from ws, into mask_votes
        np.put_along_axis(mask_votes,
                          np.expand_dims(expanded_man_votes, axis=2),
                          np.expand_dims(ws, axis=2),
                          axis=2)

        # now mask_votes has bitmasked votes with weights in them, so we can
        # sum each up, and then argmax them and this includes all the out-of-dist stuff too
        summed_votes = np.sum(mask_votes, axis=1)
        res = np.argmax(summed_votes, axis=1)
        return res






def test_detector(detector,
                  test_set,
                  broken_u_idx,
                  num_samples,
                  len_samples = 1,
                  test_forward_only=False,
                  desc=""):
    detection_dists = []
    detection_xdots = []
    detection_us = []

    false_negative_dists = []
    false_negative_xdots = []
    false_negative_us = []

    if test_forward_only:
        forward_inds = test_set.xdot[:,0]>0
        test_set_xdot = test_set.xdot[forward_inds]
        test_set_u = test_set.u[forward_inds]
        print("Only forward samples {}->{}".format(len(test_set.xdot), len(test_set_xdot)))
    else:
        test_set_xdot = test_set.xdot
        test_set_u = test_set.u

    dists = []
    with tqdm(total=num_samples) as pbar:
        for i in range(num_samples):
            # pick a random broken point to query for
            rand_indx = np.random.randint(len(test_set_xdot)-1)

            step = 1
            s = max(0, rand_indx-(len_samples*step))
            query_xdots = test_set_xdot[s:rand_indx:step]
            # the query moment is _now_
            # but we will query for some time in the past too
            query_u = test_set_u[rand_indx]
            query_xdot = test_set_xdot[rand_indx]

            dist, expected_u, neighbor_indices = detector.query(query_xdots, query_u)
            dists.append(dist)
            if dist != 0:
                # so this control is broken, but it wasnt USED for this sample anyways
                # the distance should have been 0
                # thus this is a false negative
                if query_u[broken_u_idx] == 0:
                    false_negative_dists.append(dist)
                    false_negative_xdots.append(query_xdot)
                    false_negative_us.append(query_u)
                else:
                    detection_dists.append(dist)
                    detection_xdots.append(query_xdot)
                    detection_us.append(query_u)

            pbar.update(1)

    detection_dists = np.array(detection_dists)
    detection_xdots = np.array(detection_xdots)
    detection_us = np.array(detection_us)

    false_negative_dists = np.array(false_negative_dists)
    false_negative_xdots = np.array(false_negative_xdots)
    false_negative_us = np.array(false_negative_us)

    dists = np.array(dists)

    detection_percent = 100*len(detection_dists)/num_samples
    false_detections = 100*len(false_negative_dists)/num_samples
    no_neighbors = 100*np.sum(dists < 0) / num_samples

    desc += "\nDetections={}%".format(detection_percent)
    desc += "\nFalse detections={}%".format(false_detections)
    desc += "\nNo neighbors={}%".format(no_neighbors)

    return detection_percent, false_detections, desc



if __name__ == '__main__':
    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    # General idea:
    # measure xdot live. find live xdot's neighbors in known-good data.
    # find the expected controls that should be creating these xdots from the known-good data
    # compare expectation to live applied controls.
    # if similar, dynamics are as expected, otherwise something might have changed

    normal = Dataset('normal_111_11_data.json')
    detector = KDTreeDetector(normal,
                              pos_radius = 0.01,
                              ori_radius = 0)


    # a lot of samples to test the detection with
    t1_broken = Dataset('t1_broken_data.json')
    rudder_broken = Dataset('rudder_broken_data.json')
    normal_test = Dataset('normal_31_data.json')

    # run the tests parallel
    from multiprocessing import Pool

    num_samples = 500
    len_samples = 1
    test_forward_only = False
    args = [
        (detector, t1_broken, 2, num_samples, len_samples, test_forward_only, "T1 Broken"),
        (detector, rudder_broken, 0, num_samples, len_samples, test_forward_only, "Rudder Broken"),
        (detector, detector.normal, 0, num_samples, len_samples, test_forward_only, "Positive seen samples"),
        (detector, normal_test, 0, num_samples, len_samples, test_forward_only, "Positive unseen samples")
    ]

    with Pool(processes=12) as p:
        results = p.starmap(test_detector, args)

    for result in results:
        print('-'*10)
        print(result[-1])



    # sizes = 30 * detection_dists / np.max(detection_dists)
    # win, app = plot_xyz(xyz = detection_xdots[:,:3], u = detection_us, sizes = sizes)
    # win, _ = plot_xyz(xyz = detector.normal.xdot[:,:3], u = detector.normal.u, win=win)
    # pltn.gl_line3(win, detection_xdots[:,:3])

    # app.exec()





















