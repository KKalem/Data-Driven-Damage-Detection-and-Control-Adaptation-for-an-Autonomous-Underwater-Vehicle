#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

from lolo_action import Lolo
from lolo_breaker import LoloBreaker
from kdtree_voting import KDTreeDetector
from plot_data_json import plot_xyz, Dataset
from sim_to_real import Realifier
from svgp import MultiSVGP
from plot_svgp_model import model_to_pts

from toolbox import plotting as pltn

import sys
import rospy
import time
import pickle

import numpy as np
import pyqtgraph.opengl as gl


# because rqt cant plot Bools meh...
from std_msgs.msg import Float32

def make_crosshair(xyz, size, pointer=True):
    """
    returns a list of points such that when drawn as a line
    they make a 3D crosshair around xyz
    """
    xyz = np.array(xyz)
    pts = [
        xyz + [size, 0,0],
        xyz,
        xyz - [size, 0,0],
        xyz,
        xyz + [0, size, 0],
        xyz,
        xyz - [0, size, 0],
        xyz,
        xyz + [0,0, size],
        xyz,
        xyz - [0,0, size],
        xyz
    ]
    if pointer:
        pts += [xyz, [0,0,0]]

    return np.array(pts)






if __name__ == "__main__":
    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)
    rospy.init_node("live_detection")

    # lolo can control and has 'xdot'
    lolo = Lolo()
    # breaker listens to externally applied controls, has 'u'
    # just dont run breaker.update()
    breaker = LoloBreaker()

    with open("realifier.pickle", 'rb+') as f:
        realifier = pickle.load(f)
    sim_realified = Dataset('normal_9_500rpm_data.json')
    sim_realified.xdot[:,:3] = realifier.sim_xformed

    kdtree_detector = KDTreeDetector(sim_realified, pos_radius=0.01, ori_radius=0)
    mSVGP = MultiSVGP(name_root='svgp_model_man', num_models=9)

    pts, colors, Z = model_to_pts(mSVGP, n=50)

    win, app = pltn.make_opengl_fig(grid_scale = (0.1,0.1,0.1))
    train_scatter = plot_xyz(xyz=pts, win=win, colors = colors)

    # and then plot a big ass line to a moving point
    live_line = pltn.gl_line3(w=win, pts=make_crosshair([0,0,0], 0.1))
    realified_line = pltn.gl_line3(w=win, pts=make_crosshair([0,0,0], 0.1))


    hz = 3

    crosshair_size = 0.05

    def update(timer_event):
        last_t = timer_event.last_real
        if last_t is None:
            print("last_t is none")
            return
        current_t = timer_event.current_real
        if current_t is None:
            print("current_t is none")
            return

        delta_time = current_t - last_t
        delta_time = delta_time.secs + (delta_time.nsecs * 1e-9)

        lolo.update_tf(delta_time = delta_time)

        live_u = breaker.u
        live_xdot = lolo.xdot[:3]

        realified_xdot = realifier.xform(live_xdot)[0]

        live_line.setData(pos=make_crosshair(live_xdot, crosshair_size))
        realified_line.setData(pos=make_crosshair(realified_xdot, crosshair_size*2))

        # limits found by search on extra weight dataset
        gp_pY = mSVGP.detect(realified_xdot, limit=0.9616)
        kd_pY = kdtree_detector.detect(realified_xdot, limit=0.0022)
        print(f"gp:{gp_pY}, kd:{kd_pY}")


        if rospy.is_shutdown():
            print('Rospy shutdown')
            sys.exit(0)


    t = rospy.Timer(rospy.Duration(1./hz), update)
    app.exec()


