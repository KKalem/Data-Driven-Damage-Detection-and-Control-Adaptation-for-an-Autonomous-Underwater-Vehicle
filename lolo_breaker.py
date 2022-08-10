#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import rospy
import sys

from cola2_msgs.msg import Setpoints
from std_msgs.msg import Float32
from smarc_msgs.msg import ThrusterRPM

class LoloBreaker(object):
    def __init__(self):
        self.elevator = None
        self.rudder = None
        self.thruster1 = None
        self.thruster2 = None

        self.break_elevator = False
        self.break_rudder = False
        self.break_t1 = False
        self.break_t2 = False

        self.fin_msg = Setpoints()
        self.thr_msg = Setpoints()

        self.fin_msg.setpoints = [0.]*5
        self.thr_msg.setpoints = [0.]*2

        # 0, 3, 4 are elevator, 1,2 are rudders
        self.fin_pub = rospy.Publisher('/lolo/sim/rudder_setpoints', Setpoints, queue_size=1)
        # 0 is t1, 1 is t2
        self.thruster_pub = rospy.Publisher('/lolo/sim/thruster_setpoints', Setpoints, queue_size=1)

        # we need the commands to _NOT_ break some parts...
        self.elevator_sub = rospy.Subscriber('/lolo/core/elevator_cmd', Float32, self.elev_cb)
        self.rudder_sub = rospy.Subscriber('/lolo/core/rudder_cmd', Float32, self.rudder_cb)
        self.t1_sub = rospy.Subscriber('/lolo/core/thruster1_cmd', ThrusterRPM, self.thruster1_cb)
        self.t2_sub = rospy.Subscriber('/lolo/core/thruster2_cmd', ThrusterRPM, self.thruster2_cb)

        # rudder, eleve, t1, t2 commands
        self.u = [0,0,0,0]


    def rudder_cb(self, msg):
        self.rudder = msg.data
        self.u[0] = self.rudder

        if self.break_rudder:
            self.rudder = 0

        self.fin_msg.header.stamp = rospy.Time().now()
        self.fin_msg.setpoints[1] = self.rudder
        self.fin_msg.setpoints[2] = self.rudder

    def elev_cb(self, msg):
        self.elevator = msg.data
        self.u[1] = self.elevator

        if self.break_elevator:
            self.elevator = 0

        self.fin_msg.header.stamp = rospy.Time().now()
        self.fin_msg.setpoints[0] = self.elevator
        self.fin_msg.setpoints[3] = self.elevator
        self.fin_msg.setpoints[4] = self.elevator

    def thruster1_cb(self, msg):
        self.thruster1 = msg.rpm
        self.u[2] = self.thruster1

        if self.break_t1:
            self.thruster1 = 0

        self.thr_msg.header.stamp = rospy.Time().now()
        self.thr_msg.setpoints[0] = self.thruster1/2000.

    def thruster2_cb(self, msg):
        self.thruster2 = msg.rpm
        self.u[3] = self.thruster2

        if self.break_t2:
            self.thruster2 = 0

        self.thr_msg.header.stamp = rospy.Time().now()
        self.thr_msg.setpoints[1] = self.thruster2/2000.


    def update(self):
        self.fin_pub.publish(self.fin_msg)
        self.thruster_pub.publish(self.thr_msg)


if __name__ == '__main__':
    rospy.init_node("lolo_breaker")
    breaker = LoloBreaker()

    if 'elev' in sys.argv:
        print("Breaking elevator!")
        breaker.break_elevator = True
    else:
        print("Elevator left alone")

    if 'rudder' in sys.argv:
        print("Breaking rudder!")
        breaker.break_rudder = True
    else:
        print("Rudder left alone")

    if 't1' in sys.argv:
        print("Breaking thruster 1!")
        breaker.break_t1 = True
    else:
        print("Thruster 1 left alone")

    if 't2' in sys.argv:
        print("Breaking thruster 2!")
        breaker.break_t2 = True
    else:
        print("Thruster 2 left alone")

    # double the freq. of msg_bridge, to make sure we flood the sim
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        breaker.update()
        rate.sleep()

    print("Done")




