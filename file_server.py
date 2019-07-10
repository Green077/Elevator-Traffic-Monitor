import socket
import json
import binascii
import ssl
import time
import cv2
import pandas as pd
import numpy as np
from datetime import timedelta
from os.path import dirname, join, basename
import sys
from glob import glob

file_path = 'images/'
file_path_csv = '../Desktop/timeseries.csv'
col_names = ['time', 'floor_one']
top_floor = 15
time_for_stairs = [16,16*2,16*3,16*4,16*5,16*6,16*7,16*8,16*9,16*10]
svm = cv2.ml.SVM_load('trained_model.xml')
timeseries_data = pd.read_csv(file_path_csv, header = None, names = col_names,parse_dates = [0])
print timeseries_data

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:int(x_pixel/2),:int(y_pixel/2)], bins[int(x_pixel/2):,:int(y_pixel/2)], bins[:int(x_pixel/2),int(y_pixel/2):], bins[int(x_pixel/2):,int(y_pixel/2):]
    mag_cells = mag[:int(x_pixel/2),:int(y_pixel/2)], mag[int(x_pixel/2):,:int(y_pixel/2)], mag[:int(x_pixel/2),int(y_pixel/2):], mag[int(x_pixel/2):,int(y_pixel/2):]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist

def compute_speed(p_source_floor, p_target_floor):
  final_result_index = []
  time_vertex = []
  final_result_time = []
  for x in timeseries_data.loc[timeseries_data['floor_one'] == p_source_floor].index.values:
    more_than_x_index = filter(lambda y: y > x, timeseries_data.loc[timeseries_data['floor_one'] == p_target_floor].index.values)
    if more_than_x_index != []:
      final_result_index.append(min(more_than_x_index - x))
      time_x = timeseries_data['time'][x]
      for z in more_than_x_index:
        time_more_x = timeseries_data['time'][z]
        time_vertex.append(time_more_x - time_x)
      final_result_time.append(min(time_vertex))
  final_result_index = np.asarray(final_result_index)
  final_result_time = np.asarray(final_result_time)
  ave_speed = np.mean(final_result_time/final_result_index)
  print 'ave_speed is %s' % ave_speed
  return ave_speed

def calculate_time(p_source_floor, p_target_floor,e_current_floor, up_or_down):
  ave_speed = compute_speed(p_source_floor, p_target_floor)
  time = 0
  if p_target_floor < p_source_floor:
    if up_or_down == True:
      if e_current_floor < p_source_floor:
        time = ave_speed * (p_source_floor - e_current_floor + p_source_floor - p_target_floor)
      else:
        time = ave_speed * (top_floor - e_current_floor + top_floor - p_target_floor)
    elif e_current_floor < p_source_floor:
        time = ave_speed * (e_current_floor + p_source_floor + p_source_floor - p_target_floor)
    else:
        time = ave_speed * (e_current_floor - p_target_floor)
  if p_target_floor > p_source_floor:
    if up_or_down == True:
      if e_current_floor < p_source_floor:
        time = ave_speed * (p_target_floor - e_current_floor)
      else:
        time = ave_speed * ((top_floor - p_source_floor) + (top_floor - e_current_floor) + (p_target_floor - e_current_floor))
    elif e_current_floor < p_source_floor:
      time = ave_speed * (e_current_floor + p_target_floor)
    else:
      time = ave_speed * (e_current_floor - p_source_floor + p_target_floor - p_source_floor)
  if p_source_floor == p_target_floor:
    print 'you are already in the right floor'
  return time

def judge_elevator_or_stairs(p_source_floor, p_target_floor, e_current_floor, up_or_down):
  time_elevator = calculate_time(p_source_floor, p_target_floor, e_current_floor, up_or_down)
  L = timedelta(seconds = abs(time_for_stairs[p_target_floor - 1] - time_for_stairs[p_source_floor - 1]))
  print 'time for elevator:%s'  % time_elevator
  print 'time for stairs %s' % L
  if time_elevator <= L:
    print 'choose elevator'
    elevator = 'elevator'
  else:
    print 'choose stairs'
    elevator = 'stairs'
  return elevator

def create_final_table(e_current_floor, up_or_down):
  methods = []
  for i in range(2,11):
    choice = judge_elevator_or_stairs(1,i,e_current_floor, up_or_down)
    methods.append(choice)
  return methods

def refresh_index(final_predict, up_or_down):
    method = create_final_table(final_predict, up_or_down)
    with open("./index.html","w") as f:
        f.write('<!doctype html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<title>Group7</title>\n')
        f.write('</head>\n')
        f.write('<body bgcolor="powderblue" style="font-size:20">\n')
        f.write('<h1 align="center">EECS 4764 ELEVATOR ADVISOR</h1>\n')
        f.write('<p style="font-size:20px;text-align:center">How long it will take to go to each floor:<p>\n')
        f.write('<table align="center" border="1">\n<tr>\n')
        f.write('<td align="center">'+"DESTINATION"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+"METHOD"+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp2"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[0]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp3"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[1]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp4"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[2]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp5"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[3]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp6"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[4]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp7"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[5]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp8"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[6]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp9"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[7]+"</td>\n")
        f.write('</tr>\n</tr>')
        f.write('<td align="center">'+"\t&nbsp10"+"</td>\n")
        f.write('<td>'+"-"+"</td>\n")
        f.write('<td align="center">'+method[8]+"</td>\n")
        f.write('</tr>\n')
        f.write('</body>\n')
        f.write('</html>\n')
# initialize server
# code modified from HTTP Server Example on:
# https://docs.micropython.org/en/latest/esp8266/esp8266/tutorial/network_tcp.html#simple-http-server
addr = socket.getaddrinfo('0.0.0.0', 8080)[0][-1]
s = socket.socket()
s.bind(addr)
s.listen(1)
save_flag = False
status = 0
print('Initialized File Server')

while True:
        cl, addr = s.accept()
        #print('client connected from', addr)
        cl_file = cl.makefile('rwb', 0)
        cl_file.readline()
        cl_file.readline()
        cl_file.readline()
        cl_file.readline()
        while True:
                line = cl_file.readline()
                if line == b'END IMAGE TRANSFER':
                        save_flag = False

                        # Stats
                        end_time = time.time()
                        diff = end_time-start_time
                        print('time elapsed: ' + str(diff) + 's')
                        print('packets/second: ' + str(packet_count / diff))
                        print('bytes/second: ' + str(payload_bytes / diff))
                        print('Filename: ' + fn)
                        test_temp=[]
                        for n in glob(join(dirname(__file__), '*.jpg')):
                            img=cv2.imread(n,0)
                            test_temp.append(img)
                        hogdata = map(hog,test_temp)
                        testData = np.float32(list(hogdata)).reshape(-1,bin_n*4)
                        result = svm.predict(testData)
                        print ('result', result)
                        if result > 10:
                            up_or_down = False
                            final_predict = result[0] - 10
                        else:
                            up_or_down = True
                            final_predict = result[0]
                        refresh_index(final_predict, up_or_down)
                if save_flag:
                        #print('saving payload')
                        f = open(fn, 'ab')

                        i = 0
                        l = len(line)
                        for i in range(l):
                                #print('writing byte: ' + str(line[i][0]))
                                f.write(bytes(line[i][0]))
                        f.close()
                        # Stats
                        packet_count = packet_count + 1
                        payload_bytes = payload_bytes + l
                if line == b'INIT IMAGE TRANSFER':
                        save_flag = True
                        cur_time = time.localtime()
                        fn = file_path + str(cur_time.tm_year) + "-" + str(cur_time.tm_mon) + "-" + str(cur_time.tm_hour) + "-" + str(cur_time.tm_min) + "-" + str(cur_time.tm_sec) + ".jpg"
                        print('~~~~~~~~~~~~~~~~~~~~')
                        print('Receiving File...')
                        # Stats
                        packet_count = 0
                        payload_bytes = 0
                        start_time = time.time()
                if not line:
                        break
                status = 1
        HTTP_HEADER = 'HTTP/1.1 ' + ('200' if (status == 1) else '400') + '\r\n'
        # build a response
        response = HTTP_HEADER
        # send response and close socket
        cl.send(response)
        cl.close()
