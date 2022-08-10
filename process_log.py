#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import numpy as np
import os
import json
import sys
import time
from glob import glob


from tf.transformations import euler_from_quaternion, quaternion_from_euler, \
                               quaternion_matrix, quaternion_from_matrix
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def make_relative_tf(current_pose, next_pose):
    """
    return a (x,y,z,x,y,z,w) tuple that describes the change
    from current_pose to next_pose
    source:
    https://stackoverflow.com/questions/55150211/how-to-calculate-relative-pose-between-two-objects-and-put-them-in-a-transformat
    """
    def PoseStamped_2_mat(p):
        q = p.pose.orientation
        pos = p.pose.position
        T = quaternion_matrix([q.x,q.y,q.z,q.w])
        T[:3,3] = np.array([pos.x,pos.y,pos.z])
        return T

    def Mat_2_posestamped(m,f_id=""):
        q = quaternion_from_matrix(m)
        p = PoseStamped(header = Header(frame_id=f_id),
                        pose=Pose(position=Point(*m[:3,3]),
                        orientation=Quaternion(*q)))
        return p

    def T_inv(T_in):
        R_in = T_in[:3,:3]
        t_in = T_in[:3,[-1]]
        R_out = R_in.T
        t_out = -np.matmul(R_out,t_in)
        return np.vstack((np.hstack((R_out,t_out)),np.array([0, 0, 0, 1])))

    # pose after the control is applied
    p1_xyz = next_pose[:3]
    p1_ori = next_pose[3:]
    p_o1 = PoseStamped(
        header = Header(frame_id=""),
        pose=Pose(position=Point(*p1_xyz),
                  orientation=Quaternion(*p1_ori)))

    Tw1 = PoseStamped_2_mat(p_o1)

    # pose when control was applied
    p2_xyz = current_pose[:3]
    p2_ori = current_pose[3:]
    p_o2 = PoseStamped(
        header = Header(frame_id=""),
        pose=Pose(position=Point(*p2_xyz),
                  orientation=Quaternion(*p2_ori)))

    Tw2 = PoseStamped_2_mat(p_o2)

    # we want the pose of after in the frame of during
    T2w = T_inv(Tw2)
    T21 = np.matmul(T2w, Tw1)

    p21 = Mat_2_posestamped(T21)
    p21_posi = p21.pose.position
    p21_ori = p21.pose.orientation
    q = (p21_ori.x, p21_ori.y, p21_ori.z, p21_ori.w)
    p21_ori_rpy = euler_from_quaternion(q) # because this doesnt take a Quaternion...
    p21_x = (p21_posi.x,
             p21_posi.y,
             p21_posi.z,
             p21_ori_rpy[0],
             p21_ori_rpy[1],
             p21_ori_rpy[2])

    return p21_x



def plot_path(tf_trace, equalize=True):
    pos_trace = tf_trace[:,:3]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    origin = pos_trace[0]
    pos_trace -= origin
    ax.plot(pos_trace[:,0], pos_trace[:,1], pos_trace[:,2], c='green')
    ax.text(pos_trace[0,0], pos_trace[0,1], pos_trace[0,2], "S")
    ax.text(pos_trace[-1,0], pos_trace[-1,1], pos_trace[-1,2], "E")

    if equalize:
        set_axes_equal(ax)



if __name__ == "__main__":
    try:
        __IPYTHON__
        plt.ion()
        # set suppress to False to get back scientific notation
        # precision=3 milimeters are enough...
        np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
    except:
        pass


    if len(sys.argv) < 2:
        print("Usage: ./process_log [[index|filename]*|folder]")
        sys.exit(0)


    #%run main.py 2022-01-06-16-12_lolo_coverage.json
    if '/' in sys.argv[1] or '~' in sys.argv[1]:
        print("Loading all jsons from folder")
        filename = sys.argv[1]
        filenames = glob(os.path.expanduser(filename)+"/*.json")
        filenames.sort()
    else:
        filenames = []
        for filename in sys.argv[1:]:
            if not '.json' in filename:
                print("Loading with index")
                files = glob(os.path.expanduser("~/MissionLogs")+"/*.json")
                files.sort()
                filename = files[int(filename)]
                filenames.append(filename)
            else:
                print("Loading with filename")
                filename = os.path.join(os.path.expanduser("~"), "MissionLogs", filename)
                filenames.append(filename)


    if input(f"Process these?[Y to proceed]\n{filenames}\n") != 'Y':
        sys.exit(0)

    datas = []
    for filename in filenames:
        print("Loading json {}".format(filename))
        t0 = time.time()
        with open(filename, 'r') as f:
            data = json.load(f)
        print("Loaded in {}s".format(int(time.time()-t0)))
        datas.append(data)


    # if more than one file was passed, merge them all
    data = datas[0]
    for d in datas[1:]:
        data['time_trace'].extend(d['time_trace'])
        data['tf_trace'].extend(d['tf_trace'])
        data['vehicle_data']['rudder_trace'].extend(d['vehicle_data']['rudder_trace'])
        data['vehicle_data']['elevator_trace'].extend(d['vehicle_data']['elevator_trace'])
        data['vehicle_data']['thruster1_trace'].extend(d['vehicle_data']['thruster1_trace'])
        data['vehicle_data']['thruster2_trace'].extend(d['vehicle_data']['thruster2_trace'])


    # somehow some of these tfs are shorter?
    # just 0 them out
    for i in range(len(data['tf_trace'])):
        if len(data['tf_trace'][i]) != 7:
            data['tf_trace'][i] = [0.]*7


    # the timestamp of each row in seconds
    times = np.array(data['time_trace']).astype(float)
    # x,y,z position and x,y,z,w orientation per row
    tf_trace = np.array(data['tf_trace']).astype(float)
    # rudder angles in radians, + is vehicle turning right 
    rudder_trace = np.array(data['vehicle_data']['rudder_trace']).astype(float)
    # elevator angles in radians
    elevator_trace = np.array(data['vehicle_data']['elevator_trace']).astype(float)
    # thrusters, + is forward
    thruster1_trace = np.array(data['vehicle_data']['thruster1_trace']).astype(float)
    thruster2_trace = np.array(data['vehicle_data']['thruster2_trace']).astype(float)


    # filter out nans from everything
    # first put everything into one large array
    all_traces = np.block([[times],
                           [rudder_trace],
                           [elevator_trace],
                           [thruster1_trace],
                           [thruster2_trace]]).transpose()

    all_traces = np.block([tf_trace, all_traces])
    nan_rows = np.isnan(all_traces).any(axis=1)
    clean_rows = all_traces[~nan_rows]

    # also filter out all-0 controls, useless data
    # not useless, should be used to detect external effects and such
    # allzero_u_rows = np.all(clean_rows[:,8:]== (0,0,0,0), axis=1)
    # clean_rows = clean_rows[~allzero_u_rows]


    # and then separate 
    x = clean_rows[:,:7]
    t = clean_rows[:,7]
    # skip the time column for the u
    u = clean_rows[:,8:]
    # x_i = [x,y,z, x,y,z,w]
    # u_i = [rudder, elevator, thruster1, thruster2]


    print("Generating xdot for {} points".format(len(x)))
    xdot = []
    xs = []
    us = []
    ts = []
    t0 = time.time()
    num_skipped_for_time_small = 0
    num_skipped_for_time_large = 0
    num_skipped_for_fields = 0
    for i in range(len(x)-1):
        delta_time = t[i+1] - t[i]
        # this skips the merging problem of multiple logs
        if delta_time > 10:
            num_skipped_for_time_large += 1
            continue

        if delta_time <= 0:
            num_skipped_for_time_small += 1
            continue

        relative_tf = np.array(make_relative_tf(x[i], x[i+1]))

        # sanity check, there really should be no displacement more than 10m...ever
        if any([abs(field) > 10 for field in relative_tf]):
            num_skipped_for_fields += 1
            continue

        # divide by time to get proper measures and also account
        # for possibly different frequencies of data collection
        xdot.append(relative_tf/delta_time)
        us.append(u[i])
        ts.append(t[i])
        xs.append(x[i])

    print("Skipped L:{}, S:{} points for time".format(num_skipped_for_time_large, num_skipped_for_time_small))
    print("Skipped {} points for fields".format(num_skipped_for_fields))

    xdot = np.array(xdot)
    x = np.array(xs)
    u = np.array(us)
    t = np.array(ts)
    assert len(xdot) == len(u)
    assert len(xdot) == len(t)
    assert len(xdot) == len(x)
    print("Done in {}s".format(int(time.time()-t0)))

    # finally, save this processed file into here for easier plotting etc later. 
    # no need to re-process all the time
    # im not gonna bother with fancy names and such for now...
    data = {'x':x.tolist(),
            'xdot':xdot.tolist(),
            'u':u.tolist(),
            't':t.tolist(),
            'fields':'xdot_i=[x,y,z, r,p,y] m or radians /s, u_i=[rudder,elev,t1,t2] radians and rpm',
            'source_log':filenames}

    print("Saving json")
    with open('data.json', 'w+') as f:
        json.dump(data, f)
    print("Saved data.json")
    sys.exit(0)







