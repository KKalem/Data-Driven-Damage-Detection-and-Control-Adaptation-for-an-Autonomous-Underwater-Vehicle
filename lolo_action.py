#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8


import numpy as np

import actionlib
import rospy
import tf
import time

from smarc_msgs.msg import GotoWaypointFeedback, GotoWaypointResult, GotoWaypointAction, ThrusterRPM
from std_msgs.msg import Float32, Int32

from process_log import make_relative_tf



class Vehicle(object):
    def __init__(self,
                 robot_name = 'lolo',
                 utm_link='utm',
                 base_link='base_link'):
        self.position = np.array([0,0,0], dtype=float)
        self.orientation_quat = np.array([0,0,0,0], dtype=float)
        self.orientation_rpy = np.array([0,0,0], dtype=float)

        self.xdot = np.array([0,0,0,0,0,0], dtype=float) # x,y,z,r,p,y m/s or radians/s

        # updated here
        self.speed = np.zeros_like(self.position)
        self.accel = np.zeros_like(self.speed)

        # updated by vehicle-specific subs
        self.vbs_fb = 0

        self.robot_name = robot_name
        self.utm_link = utm_link
        self.base_link = "/{}/{}".format(self.robot_name, base_link)

        self.tf_listener = tf.TransformListener()
        print("Waiting for xform listener...")
        try:
            timeout = 60
            self.tf_listener.waitForTransform(self.utm_link, self.base_link, rospy.Time(), rospy.Duration(timeout))
            print("...Got it")
        except:
            print("Couldn't get a xfrom between {} and {}".format(self.utm_link, self.base_link))


        self.running_action_index = 0
        self.running_action_begin_time = -1
        self.actions = []

        self.feedback_message = ""


    def __repr__(self):
        return "<{} xdot:{} acc:{} rpy:{}>".format(self.robot_name, self.xdot, self.accel, self.orientation_rpy)


    def update_tf(self, delta_time):
        try:
            trans, ori = self.tf_listener.lookupTransform(self.utm_link,
                                                          self.base_link,
                                                          rospy.Time(0))
        except:
            print("Couldnt get transform in update?")

        # just over-write the existing values, dont re-make an array
        # this would allow us to easily visualize real-time later, maybe
        # the [:] does exactly this, the object references stay constant this way
        old_pos = np.copy(self.position)
        old_speed = np.copy(self.speed)
        old_ori_quat = np.copy(self.orientation_quat)
        old_xdot = np.copy(self.xdot)

        self.position[:] = trans[:]
        self.speed[:] = (self.position[:] - old_pos[:]) / delta_time
        self.accel[:] = (self.speed[:] - old_speed[:]) / delta_time

        self.orientation_quat[:] = ori[:]

        euler = tf.transformations.euler_from_quaternion(ori)
        self.orientation_rpy[:] = euler[:]


        old_x = np.hstack([old_pos, old_ori_quat])
        new_x = np.hstack([self.position, self.orientation_quat])

        xdot = np.array(make_relative_tf(old_x, new_x))/delta_time
        self.xdot[:] = xdot[:]



    def add_action(self, func, args, time):
        if type(args) != tuple and type(args) != list:
            args = [args]

        print("Added {} for {}s".format(func.__name__, int(time)))
        self.actions.append((func, args, time))

    def start_actions(self):
        self.running_action_index = 0
        self.running_action_begin_time = -1

    def update_actions(self):
        """
        return True if done with the list of actions
        """

        if len(self.actions) <= 0 or self.running_action_index == -1:
            self.feedback_message = "Not started/no actions"
            return True

        if self.running_action_index >= len(self.actions):
            self.feedback_message = "Out of actions"
            return True

        func, args, duration = self.actions[self.running_action_index]

        # run the action
        elapsed = time.time() - self.running_action_begin_time
        self.feedback_message = "Running {}{}, for {} more seconds ({} of {})".format(func.__name__, args, int(duration-elapsed), self.running_action_index, len(self.actions))
        func_is_done = func(*args)

        # done with this, run the next
        if elapsed >= duration:
            self.running_action_index += 1
            self.running_action_begin_time = time.time()
            self.feedback_message = "Set new action, time"
            return False

        # maybe the function is done before the given time
        if func_is_done:
            self.running_action_index += 1
            self.running_action_begin_time = time.time()
            self.feedback_message = "Set new action, done"

        return False


    def no_accel(self, *args):
        """
        Useful when we want to "wait until the vehicle stops"
        Use accel for this rather than speed, because constant currents might
        keep the vehicle drifting all the time. But eventually a steady speed will
        be reached.
        Can be used like a vehicle action
        """
        e = 1e-3
        accel = np.linalg.norm(self.accel)
        self.feedback_message = "Waiting for no accel:{:.3f}".format(abs(accel))
        return abs(accel) <= e


    def no_z_velocity(self, *args):
        """
        Useful when we want to "wait until vehicle is neutrally bouyant"
        Can be used like a vehicle action
        """
        e = 1e-4
        z_vel = self.vehicle.speed[2]
        self.feedback_message = "Waiting for no z vel:{:.2f}".format(abs(z_vel))
        return abs(z_vel) <= e


    def get_neutral(self, vbs_action, *args):
        """
        Control the VBS tank so that the vehicle is neutrally bouyant.
        vbs_action should be a vehicle-specific function that sets the
        specific vehicles vbs tank that we are allowed to control.
        lolo should use the vbs_front and sam has only one vbs tank
        """
        #XXX: hardcoded values for now. maybe a controller later.
        if 'lolo' in self.robot_name:
            vbs_target = 94.5 # slightly bouyant
        elif 'sam' in self.robot_name:
            vbs_target = 30 #XXX untested
        else:
            print("Unknown vehicle?")
            return False

        diff = abs(self.vbs_fb - vbs_target)
        # wait until the vbs feedback is where we commanded it to be
        if diff > 0.1:
            vbs_action(vbs_target)
            self.feedback_message = "Waiting for vbs, diff:{}".format(int(diff))
            return False

        return True


    def dive_until_depth_then_neutral(self, vbs_action, z):
        """
        wait until given depth is reached, doesnt care about stability there
        most likely will overshoot
        """
        self.feedback_message = "Diving, at:{:.2f}, target:{}".format(self.position[2], z)
        if self.position[2] <= z:
            self.get_neutral(vbs_action)
            print("At depth")
            return True
        else:
            vbs_action(100)
            return False


    def rise_until_depth_then_neutral(self, vbs_action, z):
        """
        same as dive_until_depth_then_neutral, the other way
        """
        self.feedback_message = "Rising, at:{:.2f}, target:{}".format(self.position[2], z)
        if self.position[2] >= z:
            self.get_neutral(vbs_action)
            print("At depth")
            return True
        else:
            vbs_action(0)
            return False






class DynamicsCollectionAction(object):
    def __init__(self,
                 vehicle,
                 update_rate = 10,
                 action_name = 'dynamics_collection_action'):

        self.vehicle = vehicle
        self.update_rate = update_rate

        self.action_server = actionlib.SimpleActionServer(
            name = '/{}/ctrl/{}'.format(self.vehicle.robot_name, action_name),
            ActionSpec = GotoWaypointAction,
            execute_cb = self.execute_cb,
            auto_start = False)

        print("Action server started")
        self.action_server.start()

        self.feedback_message = ""
        self.feedback = GotoWaypointFeedback()

        self.is_done = False

    def on_step(self):
        """
        For every update tick.
        Most of the work should be done here.
        Return False if there is trouble.
        """
        self.feedback_message = "{};\n{}".format(self.vehicle, self.vehicle.feedback_message)
        self.is_done = self.vehicle.update_actions()

        return True

    def on_preempt(self):
        """
        Specifically when the BT sends a stop signal
        """
        print("Preempted")
        self.vehicle.reset_all()
        self.vehicle.start_actions()
        return

    def on_stop(self):
        """
        Whenever the server needs to stop, this function is called.
        """
        print("Stopped")
        result = GotoWaypointResult()
        result.reached_waypoint = False
        self.action_server.set_preempted(result, "Stopped")
        self.vehicle.reset_all()
        self.vehicle.start_actions()


    def on_done(self):
        """
        When the server is DONE, without failure or preempts
        """
        print("Done")
        result = GotoWaypointResult()
        result.reached_waypoint = True
        self.action_server.set_succeeded(result=result)
        # reset the action list, basically allow the action to repeat
        # the whole list if the BT wants to
        self.vehicle.start_actions()
        self.vehicle.reset_all()


    def execute_cb(self, goal):
        """
        The general structure of an action server, so that we dont have to fudge around
        with ros-related BS later.
        """
        rate = rospy.Rate(self.update_rate)
        print("Executing")
        self.is_done = False
        while not rospy.is_shutdown():
            self.vehicle.update_tf(delta_time = 1. / self.update_rate)

            if self.action_server.is_preempt_requested():
                self.on_preempt()
                break

            all_ok = self.on_step()
            if not all_ok:
                break
            else:
                self.feedback.ETA = rospy.Time(0)
                self.feedback.feedback_message = self.feedback_message
                self.action_server.publish_feedback(self.feedback)

            if self.is_done:
                break

            rate.sleep()

        if self.is_done:
            self.on_done()
        else:
            self.on_stop()




# class SAM(Vehicle):
    # def __init__(self):
        # super(SAM, self).__init__(robot_name = 'sam')



class Lolo(Vehicle):
    def __init__(self):
        super(Lolo, self).__init__(robot_name = 'lolo')


        # radians, +-0.6
        self.abs_rudder_limit = 0.6
        self.abs_elevator_limit = 0.6
        self.rudder_pub = rospy.Publisher('/lolo/core/rudder_cmd', Float32, queue_size=1)
        self.elevator_pub = rospy.Publisher('/lolo/core/elevator_cmd', Float32, queue_size=1)
        # rpm, +-2000
        # 1 -> right thruster
        self.abs_rpm_limit = 2000
        self.rpm1_msg = ThrusterRPM()
        self.rpm2_msg = ThrusterRPM()
        self.thruster1_pub = rospy.Publisher('/lolo/core/thruster1_cmd', ThrusterRPM, queue_size=1)
        self.thruster2_pub = rospy.Publisher('/lolo/core/thruster2_cmd', ThrusterRPM, queue_size=1)
        # percent, [0-100]
        self.vbs_front_port_pub = rospy.Publisher('lolo/core/vbs_front_port_cmd', Float32, queue_size=1)
        self.vbs_front_stbd_pub = rospy.Publisher('lolo/core/vbs_front_stbd_cmd', Float32, queue_size=1)
        self.vbs_back_port_pub = rospy.Publisher('lolo/core/vbs_back_port_cmd', Float32, queue_size=1)
        self.vbs_back_stbd_pub = rospy.Publisher('lolo/core/vbs_back_stbd_cmd', Float32, queue_size=1)

        # just check the front port as a substitude, since we wont play with the back
        # and we always set both fronts to same value
        # this will likely need to change once the sim lolo is updated
        self.vbs_front_sub = rospy.Subscriber('/lolo/core/vbs_front_port_fb', Float32, self.vbs_cb)
        self.vbs_fb = 0


    def vbs_cb(self, msg):
        self.vbs_fb = msg.data

    def rudder(self, radians):
        if abs(radians) > self.abs_rudder_limit:
            print("Given rudder angle {} is too large!".format(radians))
            radians = np.sign(radians) * np.min(abs(radians), self.abs_rudder_limit)

        self.rudder_pub.publish(radians)

    def elevator(self, radians):
        if abs(radians) > self.abs_rudder_limit:
            print("Given elevator angle {} is too large!".format(radians))
            radians = np.sign(radians) * self.abs_rudder_limit

        self.elevator_pub.publish(radians)

    def thruster1(self, rpm):
        if abs(rpm) > self.abs_rpm_limit:
            print("Given rpm for T1 {} is too large!".format(rpm))
            rpm = np.sign(rpm) * self.abs_elevator_limit

        self.rpm1_msg.rpm = int(rpm)
        self.thruster1_pub.publish(self.rpm1_msg)

    def thruster2(self, rpm):
        if abs(rpm) > self.abs_rpm_limit:
            print("Given rpm for T2 {} is too large!".format(rpm))
            rpm = np.sign(rpm) * self.abs_elevator_limit

        self.rpm2_msg.rpm = int(rpm)
        self.thruster2_pub.publish(self.rpm2_msg)

    def vbs_front(self, percent):
        if percent > 100 or percent < 0:
            print("Given percent {} for vbs_front out of bounds".format(percent))
            percent = np.min(100, np.max(0, percent))

        self.vbs_front_port_pub.publish(percent)
        self.vbs_front_stbd_pub.publish(percent)

    def vbs_back(self, percent):
        if percent > 100 or percent < 0:
            print("Given percent {} for vbs_back out of bounds".format(percent))
            percent = np.min(100, np.max(0, percent))

        self.vbs_back_port_pub.publish(percent)
        self.vbs_back_stbd_pub.publish(percent)


    def vbs_both(self, percent_front, percent_back):
        self.vbs_front(percent_front)
        self.vbs_back(percent_back)


    def reset_all(self, *args):
        self.rudder(0)
        self.elevator(0)
        self.thruster1(0)
        self.thruster2(0)


    def control_all(self, rudder, elevator, thruster1, thruster2):
        self.rudder(rudder)
        self.elevator(elevator)
        self.thruster1(thruster1)
        self.thruster2(thruster2)


    def pick_random_controls(self, *args):
        angles = np.linspace(-0.6,0.6, 3)
        rpms = [500]

        self.random_rudder = np.random.choice(angles)
        self.random_elevator = np.random.choice(angles)
        self.random_thruster1 = int(np.random.choice(rpms))
        self.random_thruster2 = int(np.random.choice(rpms))
        print("Picked picked controls:{} {} {} {}".format(self.random_rudder, self.random_elevator, self.random_thruster1, self.random_thruster2))


    def control_all_randomly(self, *args):
        self.feedback_message = "Random controls: {:.2f},{:.2f},{},{}".format(self.random_rudder,
                                                                                      self.random_elevator,
                                                                                      self.random_thruster1,
                                                                                      self.random_thruster2)
        self.control_all(self.random_rudder, self.random_elevator, self.random_thruster1, self.random_thruster2)



def random_range(lim):
    return np.random.random()*(lim*2) - lim



if __name__ == "__main__":
    rospy.init_node("lolo_data_collector_action")
    np.set_printoptions(precision=4, floatmode='fixed', suppress=True)


    no_accel_wait = np.random.randint(10,300)
    print("Accel waiting time:{}".format(no_accel_wait))

    lolo = Lolo()

    def do_9():
        e_angles = np.linspace(-0.6,0.6, 3)
        r_angles = np.linspace(-0.6,0.6, 3)
        t1 = 500
        t2 = 100
        for rudder in r_angles:
            for elev in e_angles:
                lolo.add_action(lolo.reset_all, None, 1)
                lolo.add_action(lolo.dive_until_depth_then_neutral, (lolo.vbs_front, -10), 200)
                lolo.add_action(lolo.rise_until_depth_then_neutral, (lolo.vbs_front, -19), 200)
                lolo.add_action(lolo.no_accel, None, no_accel_wait)
                lolo.add_action(lolo.control_all, (rudder, elev, t1, t2), 180)


    def do_random():
        lolo.add_action(lolo.reset_all, None, 1)
        # at least be at 5m depth
        lolo.add_action(lolo.dive_until_depth_then_neutral, (lolo.vbs_front, -10), 200)
        # at most be at 15m depth
        lolo.add_action(lolo.rise_until_depth_then_neutral, (lolo.vbs_front, -19), 200)
        lolo.add_action(lolo.no_accel, None, no_accel_wait)

        # lolo.add_action(lolo.control_all, (0.6, 0, -2000, -2000), 5)
        lolo.add_action(lolo.pick_random_controls, None, 1)
        lolo.add_action(lolo.control_all_randomly, None, 180)


        lolo.add_action(lolo.reset_all, None, 1)
        lolo.start_actions()


    do_9()
    # do_random()

    action = DynamicsCollectionAction(lolo)


