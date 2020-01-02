# encoding: utf-8
from __future__ import print_function#model 溢出，10个HMM模型作平均  或者迭代训练
import numpy as np
import math

import pandas as pd
import csv
import matplotlib.pyplot as plt
global data_table
data_table = []
global data_hmm
data_hmm = []
global data_change
data_change = []
def Index_lane(data,data_lane0):
    index=np.zeros(len(data))
    data_time=data['Global_Time']
    lane0_time=data_lane0['Global_Time']

    lane_dex=0
    for i in range(len(data_time)):

        if (abs(data_time[i]-lane0_time[lane_dex])<0.5):
            index[i]=int(lane_dex)
        elif (lane_dex>=len(lane0_time)-1):
            index[i]=int(len(lane0_time)-1)
            continue
        elif (abs(data_time[i]-lane0_time[lane_dex+1])<0.5):
            lane_dex += 1
            index[i] = int(lane_dex)
        elif (data_time[i]>lane0_time[lane_dex] and data_time[i]<lane0_time[lane_dex+1]):
            index[i]=int(lane_dex)

        elif (data_time[i]>=lane0_time[lane_dex+1]):
            while (lane_dex<len(lane0_time)-1 and data_time[i]>lane0_time[lane_dex+1]):
                lane_dex+=1
            index[i]=int(lane_dex)
    return index
def poly(y):
    len_x = len(y)
    y_change = np.zeros(len_x)
    for i in range(len_x):
        if (i < 20):
            y_change[i] = y[i]
        else:
            for j in range(i - 20, i):
                y_change[i] += y[j] * 0.05
    if (len_x < 40):
        x = np.linspace(1, len_x, len_x)
        f1 = np.polyfit(x, y_change, 1)
    else:
        x = np.linspace(1, 39, 39)
        #y_select = np.zeros(20)
        y_select = y_change[len_x - 40:len_x - 1]
        f1 = np.polyfit(x, y_select, 1)
    p1 = np.poly1d(f1)
    L = 0
    for i in range(20, len_x):
        L = L + pow(y[i] - p1(i), 2)
    if (L>0.16):
        curv=0
    else:
        curv = float(f1[0] * 20)
    return curv
def sample_point(lane_index):
    lane_index = lane_index.replace('[', '')
    lane_index = lane_index.replace(']', '')
    lane1 = lane_index.split(',')
    f = np.zeros((200, 2))
    c0 = float(lane1[0])
    c1 = float(lane1[1])
    c2 = float(lane1[2])
    c3 = float(lane1[3])
    # a=state_x
    #print (c0)
    #print (type(c0))
    f[0][0] = 0
    f[0][1] = c0
    for i in range(1, 200, 1):
        f[i][0] = f[i - 1][0] + 0.5
        f[i][1] = c0 + c1 * f[i][0] + c2 * f[i][0] ** 2 + c3 * f[i][0] ** 3
    return f,c0,c1,c2,c3
def frenet(x1, y1, m,c0,c1,c2,c3):
    loc1 = 0
    loc2 = len(m) - 1
    ab=1
    distance_min = float('inf')
    distance_ori_min=float('inf')
    for i in range(len(m)):
        point = m[i]
        distance = (point[0] - x1) ** 2 + (point[1] - y1) ** 2
        if (distance < distance_min):
            loc1 = i
            distance_min = distance

    dis_online=y1-(c0 + c1 * x1 + c2 * x1 ** 2 + c3 * x1 ** 3)

    if (loc1 > 0.1 and loc1 < len(m) - 1 and abs(dis_online)>0.1):
        point1 = m[loc1 - 1]
        point2 = m[loc1 + 1]
        if (((point1[0] - x1) ** 2 + (point1[1] - y1) ** 2) > ((point2[0] - x1) ** 2 + (point2[1] - y1) ** 2)):
            loc2 = loc1 + 1
        else:
            loc2 = loc1 - 1
        ab = ((m[loc1][0] - m[loc2][0]) ** 2 + (m[loc1][1] - m[loc2][1]) ** 2) ** 0.5
        s = abs((x1 - m[loc1][0]) * (m[loc2][1] - m[loc1][1]) - (y1 - m[loc1][1]) * (m[loc2][0] - m[loc1][0]))
        dis = s / ab
        dis_to_loc1 = ((x1 - m[loc1][0]) ** 2 + (y1 - m[loc1][1]) ** 2) - dis ** 2
        vertical_x = (m[loc2][0] - m[loc1][0]) * dis_to_loc1 / ab + m[loc1][0]
        vertical_y = (m[loc2][1] - m[loc1][1]) * dis_to_loc1 / ab + m[loc1][1]
    else:
        ab = 0
        dis = y1-c0

    cross_rd_nd = y1-(c0 + c1 * x1 + c2 * x1 ** 2 + c3 * x1 ** 3)
    l = math.copysign(dis, cross_rd_nd)


    for i in range(len(m)):
        point = m[i]
        distance = (point[0]) ** 2 + (point[1] ) ** 2
        if (distance < distance_ori_min):
            loc3 = i
            distance_ori_min = distance
    if (loc3 > 0.1 and loc3 < len(m) - 1):
        point1 = m[loc3 - 1]
        point2 = m[loc3 + 1]
        if (((point1[0]) ** 2 + point1[1] ** 2) > (point2[0] ** 2 + point2[1]** 2)):
            loc4 = loc3 + 1
        else:
            loc4 = loc3 - 1
        ori_ab = ((m[loc3][0] - m[loc4][0]) ** 2 + (m[loc3][1] - m[loc4][1]) ** 2) ** 0.5
        ori_s = abs(m[loc3][0] * (m[loc4][1] - m[loc3][1]) - m[loc3][1] * (m[loc4][0] - m[loc3][0]))
        ori_dis = ori_s / ori_ab
    else:
        ori_dis = (m[loc3][0]** 2 + m[loc3][1]**2) ** 0.5


    dis_to_loc2 = (m[loc3][0] ** 2 + m[loc3][1] ** 2) - ori_dis ** 2
    s=((m[loc3][0] - m[loc1][0]) ** 2 + (m[loc3][1] - m[loc1][1]) ** 2) ** 0.5+ dis - ori_dis
    l1=y1-c0
    #print(c0)
    return s, l
def save_data(id_car, test_data):
    check = 0
    data_wei = 0
    label = 0
    del_list=[]
    global data_table
    if (len(data_table) >= 1):
        for i in range(len(data_table)):
            #print len(data_table)
            id_data = data_table[i]
            if (abs(np.array(id_data).ndim - 1) < 0.1):
                if (abs(id_car - id_data[0]) < 0.5 ):#and test_data[0][1]>0.1
                    check += 1
                    if ((np.array(test_data[0][1]) - np.array(id_data[1])) > 6000000):
                        #np.delete(data_table[i], 0, axis=0)
                        del_list.append(i)
                        break

                    else:
                        data_table[i] = np.concatenate((np.array(data_table[i]), np.array(test_data)), axis=0)
                        if (len(id_data) > 199):
                            data_table[i] = np.delete(data_table[i], [0, 60], axis=0)
                            #print 'exceed', id_car
                            break
            elif (abs(np.array(id_data).ndim - 1) > 0.5 and check<0.1):
                if ((test_data[0][1] - id_data[-1][1]) > 2000000):
                    #np.delete(data_table[i], 0, axis=0)
                    del_list.append(i)
                    continue
                if (abs(id_car - id_data[0][0]) < 0.1):# and test_data[0][1]>0.1
                    check += 1
                    data_table[i] = np.concatenate((np.array(data_table[i]), np.array(test_data)), axis=0)
                    if (len(id_data) > 199):
                        #print 'exceed', id_car
                        data_table[i]=np.delete(data_table[i], [0,60], axis=0)
                    break
        for j in range(len(del_list)):
            del data_table[del_list[-j-1]]
        for i in range(len(data_table)):
            #print len(data_table)
            id_data = data_table[i]
            if (abs(np.array(id_data).ndim - 1) < 0.1):
                if (abs(id_car - id_data[0]) < 0.1 ):#and test_data[0][1]>0.1
                    check += 1
                    label = i
                    break
            elif (abs(np.array(id_data).ndim - 1) > 0.5 ):#and check<0.1
                if (abs(id_car - id_data[0][0]) < 0.1):# and test_data[0][1]>0.1
                    check += 1
                    label = i
                    break


    if (abs(check) < 0.1):
        if (len(data_table) < 1):
            data_table.append(test_data)
        else:
            label = len(data_table)
            c = test_data
            data_table.append(c)
    return check, label

def Cut_in_label(data,data_lane1,lane_index):
    time_stamp=data['Global_Time']
    #print('cut_in_label')
    cut_label=np.zeros(len(time_stamp))
    for i in range(len(time_stamp)):
        if (abs(data['Vehicle_ID'][i]) > 0.5 and data['Valid'][i] and data['Vehicle_ID'][i] < 100):#
            curve = 0.0000001
            curve_x = 0.0000001
            id_car =data['Vehicle_ID'][i]
            time=data['Global_Time'][i]
            Local_x = data['contour_X'][i]
            Local_y = data['contour_Y'][i]
            #print ('cut')
            #lane_index = data['lane1'][i]

            if (lane_index[i]>=len(data_lane1)):
                lane_msg = data_lane1['Lane_Boundary_Right_Params'][len(data_lane1)-1]
            else:
                lane_msg=data_lane1['Lane_Boundary_Right_Params'][lane_index[i]]
            state_vx=0
            point,c0,c1,c2,c3 = sample_point(lane_msg)
            state_frenet_x, state_frenet_y = frenet(Local_x, Local_y, point,c0,c1,c2,c3)
            if (Local_x < 80 and -3.5 < state_frenet_y and state_frenet_y < 6 and Local_x > 0):  # and abs(id_car - 1381446) < 0.1
                test_data = [[id_car, time, Local_y, state_frenet_y, state_vx, Local_x,i]]
                #print (state_frenet_y,Local_y,c0)
                check, label = save_data(id_car, test_data)
                if (check > 0):
                    predict_data = data_table[label]
                    #print (predict_data)
                    #print('here111')
                    if (len(predict_data) > 20):
                        #print('here11')
                        a23 = predict_data[:, 2:]
                        num = len(a23)
                        totol_plo_y = poly(np.array(a23[num - 20:, 0]))
                        #label keep_lane
                        if (len(predict_data) > 40):
                            if (max(predict_data[:,3])<0 and min(predict_data[:,3])>-3.7 and data['Valid'][i]  and c0<0 and c0>-3.7):
                                for keep in range(len(predict_data)):
                                    id = int(predict_data[keep][6])
                                    cut_label[id] = 2
                            if (max(predict_data[:,3])<7.4 and min(predict_data[:,3])>3.75 and data['Valid'][i]  and c0<0 and c0>-3.7):
                                for keep in range(len(predict_data)):
                                    id = int(predict_data[keep][6])
                                    cut_label[id] = 2
                        #label cut_in
                        if (0.5 < state_frenet_y and state_frenet_y < 3.2 and Local_x > 0 and abs(totol_plo_y)>0.01  and data['Valid'][i] and c0<0 and c0>-3.7):  # and Local_x <100
                            #print ('here')
                            #trend = poly(np.array(a23[:, 0]))
                            if (predict_data[0][2]<-0.2 and totol_plo_y>0.01):
                                #print('herefreer')                                
                                for cut in range (num-21,0,-1):
                                    curve = poly(np.array(a23[cut:cut+20, 0]))
                                    if (curve>0.3 and abs(state_frenet_y)/abs(curve) <130 ):
                                        for j in range(cut,num):
                                            id_index=int(predict_data[j][6])
                                            cut_label[id_index] = 1
                                            #print ('cut')
                                        break
                                    
                            elif (predict_data[0][2]>4 and totol_plo_y<-0.01):
                                for cut in range (num-21,0,-1):
                                    curve = poly(np.array(a23[cut:cut+20, 0]))
                                    if (curve<-0.3 and abs((state_frenet_y-3.75)/curve )<130 ):
                                        for j in range(cut,num):
                                            id_index=int(predict_data[j][6])
                                            cut_label[id_index] = 1
                                            #print ('cut')
                                        break
    return cut_label

def cal_num_traj(data,data_lane1,cut_label,lane_index):
    dic1 = {}
    dic2 = {}
    dic = {}
    dicc = {}
    cut_num=0
    keep_num=0
    all_num=0
    ex_num=0
    time_stamp = data['Global_Time']
    #lane_index=data['lane1']
    for i in range(len(time_stamp)):
        num= int(float(lane_index[i]))
        lane_msg=data_lane1['Lane_Boundary_Right_Params'][num]
        lane_msg = lane_msg.replace('[', '')
        lane_msg = lane_msg.replace(']', '')
        lane1 = lane_msg.split(',')
        c0 = float(lane1[0])
        if (float(data['Vehicle_ID'][i]) > 0.5 and data['Valid'][i] and data['contour_X'][i]>0 and data['contour_Y'][i]-c0>-3.7 and data['contour_Y'][i]-c0<7.4 and c0<0 and c0>-3.7 and data['contour_X'][i]<100):
            time = data['Global_Time'][i]
            id_car = data['Vehicle_ID'][i]
            if (dic.get(id_car,0)>1):
                if ((dic[id_car]-time)<-2000000):
                    all_num=all_num+1
                    if (dicc[id_car]<20):
                        dicc[id_car]=1
                    else:
                        ex_num=ex_num+1
                else:
                    dicc[id_car]=dicc[id_car]+1
            else:
                dicc[id_car]= 1
            dic[id_car]=time

        if (abs(data['Vehicle_ID'][i]) > 0.5 and data['Valid'][i] and data['contour_X'][i]>0 and cut_label[i]>0.5):# and data['Vehicle_ID'][i] < 100
            if (cut_label[i]>1.5):
                id_car = data['Vehicle_ID'][i]
                time = data['Global_Time'][i]

                if (dic1.get(id_car, 0)>1):
                    if (dic1[id_car] - time<-2000000):
                        cut_num=cut_num+1
                dic1[id_car]=time
            else:
                id_car = data['Vehicle_ID'][i]
                time = data['Global_Time'][i]
                if (dic2.get(id_car, 0)>1):
                    if (dic2[id_car] - time<-2000000):
                         keep_num=keep_num+1
                dic2[id_car]=time
    cut_num=cut_num+len(dic1)
    keep_num=keep_num+len(dic2)
    all_num= all_num+len(dic)
    for value in dicc.values():
        if (value>20):
            ex_num=ex_num+1
    return cut_num,keep_num,all_num,ex_num
def save_hmm_data(id_car, test_data):
    check = 0
    data_wei = 0
    label = 0
    del_list = []
    global data_hmm
    if (len(data_hmm) >= 1):
        for i in range(len(data_hmm)):
            # print len(data_table)
            id_data = data_hmm[i]
            if (abs(np.array(id_data).ndim - 1) < 0.1):
                if (abs(id_car - id_data[0]) < 0.5):  # and test_data[0][1]>0.1
                    check += 1
                    if ((np.array(test_data[0][1]) - np.array(id_data[1])) > 6000000):
                        # np.delete(data_table[i], 0, axis=0)
                        del_list.append(i)
                        break

                    else:
                        data_hmm[i] = np.concatenate((np.array(data_hmm[i]), np.array(test_data)), axis=0)
                        if (len(id_data) > 199):
                            data_hmm[i] = np.delete(data_hmm[i], [0, 60], axis=0)
                            # print 'exceed', id_car
                            break
            elif (abs(np.array(id_data).ndim - 1) > 0.5 and check < 0.1):
                if ((test_data[0][1] - id_data[-1][1]) > 2000000):
                    # np.delete(data_table[i], 0, axis=0)
                    del_list.append(i)
                    continue
                if (abs(id_car - id_data[0][0]) < 0.1):  # and test_data[0][1]>0.1
                    check += 1
                    data_hmm[i] = np.concatenate((np.array(data_hmm[i]), np.array(test_data)), axis=0)
                    if (len(id_data) > 199):
                        # print 'exceed', id_car
                        data_hmm[i] = np.delete(data_hmm[i], [0, 60], axis=0)
                    break
        for j in range(len(del_list)):
            del data_hmm[del_list[-j - 1]]
        for i in range(len(data_hmm)):
            # print len(data_table)
            id_data = data_hmm[i]
            if (abs(np.array(id_data).ndim - 1) < 0.1):
                if (abs(id_car - id_data[0]) < 0.1):  # and test_data[0][1]>0.1
                    check += 1
                    label = i
                    break
            elif (abs(np.array(id_data).ndim - 1) > 0.5):  # and check<0.1
                if (abs(id_car - id_data[0][0]) < 0.1):  # and test_data[0][1]>0.1
                    check += 1
                    label = i
                    break

    if (abs(check) < 0.1):
        if (len(data_hmm) < 1):
            data_hmm.append(test_data)
        else:
            label = len(data_hmm)
            c = test_data
            data_hmm.append(c)
    return check, label
def HMM_label(data, data_lane1, lane_index,truelabel):
    time_stamp = data['Global_Time']
    cut_label = np.zeros(len(time_stamp))
    for i in range(len(time_stamp)):
        if (abs(data['Vehicle_ID'][i]) > 0.5 and data['Valid'][i] and data['contour_X'][i]>0):  # and data['Vehicle_ID'][i] < 100
            curve = 0.0000001
            curve_x = 0.0000001
            A = np.array([[0.8, 0.19, 0.01], [0.19, 0.8, 0.01], [0.19, 0.01, 0.8]])
            P = np.array([1, 0, 0])
            pyb = np.array([0, 0, 0])
            id_car = data['Vehicle_ID'][i]
            time = data['Global_Time'][i]
            Local_x = data['contour_X'][i]
            Local_y = data['contour_Y'][i]

            # lane_index = data['lane1'][i]

            if (lane_index[i] >= len(data_lane1)):
                lane_msg = data_lane1['Lane_Boundary_Right_Params'][len(data_lane1) - 1]
            else:
                lane_msg = data_lane1['Lane_Boundary_Right_Params'][lane_index[i]]
            state_vx = 0
            point, c0, c1, c2, c3 = sample_point(lane_msg)
            state_frenet_x, state_frenet_y = frenet(Local_x, Local_y, point, c0, c1, c2, c3)
            if (Local_x < 80 and -3.5 < state_frenet_y and state_frenet_y < 6 and Local_x > 0):  # and abs(id_car - 1381446) < 0.1
                test_data = [[id_car, time, Local_y, state_frenet_y, state_vx, Local_x]]
                check, label = save_hmm_data(id_car, test_data)
                mm = data_hmm[label]
                if (check > 0):
                    predict_data = data_hmm[label]

                    if (len(predict_data) > 20):

                        a23 = predict_data[:, 2:]
                        #a23 = exponential_smoothing(0.7, origin_a23)
                        num = len(a23)
                        # print a23[:, 0]
                        totol_plo_y = np.array(a23[num - 20:, 0])
                        totol_curve = poly(totol_plo_y)

                        if (state_frenet_y > 5 and state_frenet_y < 7 and totol_curve < 0):  # (16.25*(state_frenet_y-lane1[4])**3)+40)and abs(Local_y-lane1[4])/abs(curve) <130)
                            if (num > 20 and num < 41):
                                for i in range(num - 20):
                                    # print a23
                                    plo_y = np.array(a23[:i + 20, 0])
                                    # print len(plo_y)
                                    curve = poly(plo_y)
                                    if (curve > 0):
                                        pyb[0] = 1
                                        # print(curve)
                                    else:
                                        pyb[0] = float(math.exp(curve * 1))
                                    pyb[2] = 1 - pyb[0]
                                    P = pyb * (P.dot(A))
                                    norm = P[0] + P[1] + P[2]
                                    P = P / norm
                            else:
                                for i in range(40, num):
                                    plo_y = np.array(a23[i - 40:i, 0])

                                    curve = poly(plo_y)
                                    if (curve > 0):
                                        pyb[0] = 1
                                        # print(curve)
                                    else:
                                        pyb[0] = float(math.exp(curve * 1))
                                    pyb[2] = 1 - pyb[0]
                                    P = pyb * (P.dot(A))
                                    norm = P[0] + P[1] + P[2]
                                    P = P / norm

                        elif (state_frenet_y < 2 and state_frenet_y > -6 and totol_curve > 0):  # and abs(state_frenet_y)/abs(curve) <130
                            if (num > 20 and num < 41):
                                for i in range(num - 20):
                                    plo_y = np.array(a23[:i + 20, 0])
                                    plo_x = np.array(a23[:i + 20, 3])
                                    curve = poly(plo_y)
                                    curve_x = poly(plo_x)
                                    # print curve
                                    if (curve < 0):
                                        curve = 0
                                        pyb[0] = 1
                                    elif (curve < 0.5 and curve_x < 0):
                                        pyb[0] = 1
                                    else:
                                        pyb[0] = float(math.exp(-curve * 1))
                                    pyb[1] = 1 - pyb[0]
                                    P = pyb * (P.dot(A))
                                    norm = P[0] + P[1] + P[2]
                                    P = P / norm
                            else:
                                for i in range(40, num):
                                    plo_y = np.array(a23[i - 40:i, 0])
                                    curve = poly(plo_y)
                                    if (curve < 0):
                                        curve = 0
                                        pyb[0] = 1
                                    elif (curve < 0.5 and curve_x < 0):
                                        pyb[0] = 1
                                    else:
                                        pyb[0] = float(math.exp(-curve *2))
                                    pyb[1] = 1 - pyb[0]
                                    P = pyb * (P.dot(A))
                                    norm = P[0] + P[1] + P[2]
                                    P = P / norm
                        if (Local_x > 0 and P[0] < P[1]):  # and Local_x <100-3 < state_frenet_y and state_frenet_y < 3 and
                            cut_label[i] = 3
                        #elif (P[0] > P[1] and abs(truelabel[i]-1)<0.1):
                            #print (P[0],P[1],curve)

                        elif (6 > state_frenet_y and state_frenet_y > 3.5 and Local_x > 0 and P[0] < P[2]):
                            cut_label[i]  = 3

                        if (state_frenet_y < 3.2 and state_frenet_y > 0.5):
                            cut_label[i] = 3



    return cut_label

if __name__ == '__main__':
    in_in = 0
    in_out = 0
    out_out = 0
    out_in = 0
    final_label = np.zeros(10)
    filePath = '/home/holo/Documents/cut_in_bag/test_bag/small_sample_csv'
    site = '/1101_1_cut_in'
    data_path=filePath+site+site+'.csv'
    data_path0 = filePath +site+site+ '_lane0.csv'
    data_path1 = filePath +site+site+ '_lane1.csv'
    data_path2 = filePath +site+site+ '_lane2.csv'

    data = pd.read_csv(data_path)
    #data_lane0= pd.read_csv(data_path0)
    data_lane1= pd.read_csv(data_path1)
    lane_index =Index_lane(data,data_lane1)
    df1 = pd.DataFrame(lane_index, columns=['lane0'])
    df2 = pd.DataFrame(lane_index, columns=['lane1'])
    df3 = pd.DataFrame(lane_index, columns=['lane2'])

    data1=pd.concat([data,df1,df2,df3],axis=1)
    #data1.to_csv(r"/home/holo/Documents/clean/2019-11-14-17-16-01/new11.csv", mode='a', index=False)
    cut_label=Cut_in_label(data,data_lane1,lane_index)

    df4 = pd.DataFrame(cut_label,  columns=['cut_in'])
    #data['cut_in'] = cut_label
    result=pd.concat([data,df1,df2,df3,df4],axis=1)
    result.to_csv(filePath +site+'/cnew.csv', mode='a', index=False)
    #print(data.columns)11
    '''
    cut_num, keep_num, all_num, ex_num = cal_num_traj(data, data_lane1,cut_label,lane_index)
    print(cut_num, keep_num, all_num, ex_num)

    hmm_label = HMM_label(data, data_lane1, lane_index,cut_label)
    for i in range(len(hmm_label)):
        if (abs(cut_label[i]-1)<0.1 and abs(hmm_label[i]-3)<0.1):
            in_in=in_in+1
        if (abs(cut_label[i]-1)<0.1 and abs(hmm_label[i])<0.1):
            in_out=in_out+1
        if (abs(cut_label[i] -2) < 0.1 and abs(hmm_label[i]) < 0.1):
            out_out = out_out + 1
        if (abs(cut_label[i]-2)<0.1 and abs(hmm_label[i]-3)<0.1):
            out_in=out_in+1
    print (in_in,out_in,in_out,out_out)
    # plt.figure(num=1)
    # plt.plot(x,y1)
    final_label[0] = cut_num
    final_label[1] = keep_num
    final_label[2] = all_num
    final_label[3] = ex_num
    final_label[4] = in_in
    final_label[5] = out_in
    final_label[6] = in_out
    final_label[7] = out_out
    final_label[8] = float(in_in/(in_in+in_out))
    final_label[9] = float(in_in / (in_in+out_in))
    np.savetxt(filePath + site + 'result.csv', final_label, delimiter=',')
    x = []
    y = []
    number = 0
    count = 0
    data_y = data['contour_Y']
    data_id = data['Vehicle_ID']
    id = []
    for i in range(len(cut_label)):
        if (cut_label[i] > 0.5 and cut_label[i] < 1.3):
            x.append(number)
            number += 1
            y.append(data_y[i])
            if data_id[i] not in id:
                id.append(data_id[i])

    plt.scatter(x, y, s=2)
    x1=[]
    y1=[]
    id1=[]
    number1 = 0
    for i in range(len(hmm_label)):
        if (abs(cut_label[i]-1)<0.1 and abs(hmm_label[i])<0.1):
            x1.append(number1)
            number1 += 1
            y1.append(data_y[i])
            if data_id[i] not in id:
                id.append(data_id[i])
    for j in range(len(y1)-20):
        A = np.array([[0.8, 0.19, 0.01], [0.19, 0.8, 0.01], [0.19, 0.01, 0.8]])
        P = np.array([1, 0, 0])
        pyb = np.array([0, 0, 0])
        for i in range(j,j + 20):
            curve = poly(y1[j:j+20])
            print (curve)
            pyb[0] = float(math.exp(-curve * 1))
            pyb[1] = 1 - pyb[0]
            P = pyb * (P.dot(A))
            norm = P[0] + P[1] + P[2]
            P = P / norm
            print (P[0],P[1],pyb[0],pyb[1])
    plt.scatter(x1, y1, s=2)
    plt.ylim(-5, 25)

    plt.show()

    '''









