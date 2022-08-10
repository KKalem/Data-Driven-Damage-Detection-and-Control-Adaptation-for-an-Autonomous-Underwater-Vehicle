#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import sys, time, os, json, glob, rospy, tf

def extract_trans_rot(bbstr):
    remove_list = [
        '\x1b',
        '[33m',
        '[32m',
        '[0m',
        '[36m'
    ]

    for s in remove_list:
        bbstr = bbstr.replace(s, '')

    lines = bbstr.split('\n')
    rot = ""
    trans = ""
    for line in lines:
        if 'world_rot' in line:
            rot = eval(line.split(':')[-1].strip())

        if 'world_trans' in line:
            trans = eval(line.split(':')[-1].strip())

    return trans, rot


topics_needed=[
    '/lolo/core/elevator_cmd',
    '/lolo/core/thruster1_cmd',
    '/lolo/core/thruster2_cmd',
    '/lolo/core/rudder_cmd'
]
bb_topic = '/lolo/smarc_bt/blackboard'

# also replace the cmd/fb and add those in too...
expanded_topics = []
for topic in topics_needed:
    new = None
    if topic[-3:] == 'cmd':
        new = topic[:-3] + 'fb'

    if topic[-2:] == 'fb':
        new = topic[:-2] + 'cmd'

    if new is not None:
        expanded_topics.append([topic, new])
    else:
        expanded_topics.append([topic])



bagfiles = glob.glob('lolobags/*.bag')
# i = 0
# bagfile = bagfiles[i]

for i, bagfile in enumerate(bagfiles):
    print('*'*10)
    print(f"Reading {bagfile}, {i+1}/{len(bagfiles)}")
    print('*'*10)

    bag = rosbag.Bag(bagfile)
    print("Done")

    types, available_topics = bag.get_type_and_topic_info()

    if bb_topic not in available_topics:
        print(f"{bagfile} doesn't have blackboard in it, skip!")
        continue

    # out of the alternatives, find out which are available
    # if one is, record that one
    # if none is, this bag is useless to me
    needed_and_available_topics = []
    missing_topic = None
    for alternatives in expanded_topics:
        found = False
        for alt in alternatives:
            if alt in available_topics:
                needed_and_available_topics.append(alt)
                found = True
                break
        if not found:
            missing_topic = alternatives
            break

    if len(needed_and_available_topics) != len(topics_needed):
        print(f"{bagfile} is missing a topic")
        bag.close()
        continue

    needed_and_available_topics.append(bb_topic)


    last_tick_time = 0
    hz = 3

    rcv_data = {
        'trans':None,
        'rot':None,
        'elev':None,
        'rudder':None,
        't1':None,
        't2':None
    }

    # mimic the mission_log object
    time_trace = []
    tf_trace = []
    vehicle_data = {
        'elevator_trace':[],
        'rudder_trace':[],
        'thruster1_trace':[],
        'thruster2_trace':[]
    }

    for topic, msg, t in bag.read_messages(topics=needed_and_available_topics):

        if topic == bb_topic:
            trans, rot = extract_trans_rot(msg.data)
            rcv_data['trans'] = trans
            rcv_data['rot'] = rot

        if 'elevator' in topic:
            rcv_data['elev'] = msg.data

        if 'rudder' in topic:
            rcv_data['rudder'] = msg.data

        if 'thruster1' in topic:
            rcv_data['t1'] = msg.rpm.rpm

        if 'thruster2' in topic:
            rcv_data['t2'] = msg.rpm.rpm


        t_secs = t.to_sec()
        if t_secs - last_tick_time > 1./hz:
            last_tick_time = t_secs
            try:
                tf = []
                tf.extend(rcv_data['trans'])
                tf.extend(rcv_data['rot'])
            except:
                # just skip the times where tf is useless...
                continue
            tf_trace.append(tf)

            time_trace.append(t_secs)
            vehicle_data['elevator_trace'].append(rcv_data['elev'])
            vehicle_data['rudder_trace'].append(rcv_data['rudder'])
            vehicle_data['thruster1_trace'].append(rcv_data['t1'])
            vehicle_data['thruster2_trace'].append(rcv_data['t2'])


    bag.close()

    data = {
        'tf_trace':tf_trace,
        'time_trace':time_trace,
        'vehicle_data':vehicle_data,
    }

    json_filename = bagfile + '.json'
    with open(json_filename, 'w+') as f:
        json.dump(data, f)


print("ALL DONE")
print("ALL DONE")
print("ALL DONE")
print("ALL DONE")
print("ALL DONE")






