import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import os
import glob
import xml.etree.cElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
import operator
import pyautogui

def check_dir(dir_list):
    for d in dir_list:
        if not os.path.isdir(d):
            print('Create directory :\n' + d)
            os.makedirs(d)

def get_boxes_from_voc(file_path):
    # Get the bounding box from annotations file
    # input(file_path) : Annotation file path.
    # return(boxes_list) : List of bounding box. [box 1, box 2, ...]
    # box : [category, accuracy, [middle_x, middle_y, box width, box height]
    boxes_list = []
    xmltree = ET.ElementTree(file = file_path)
    root = xmltree.getroot()
    
    for child in root:
       if child.tag == 'object':
            name = child[0].text
            xmin = int(child[4][0].text)                    
            ymin = int(child[4][1].text)         
            xmax = int(child[4][2].text) 
            ymax = int(child[4][3].text)

            if name == 'car' or name == 'truck'or name == 'bus':
                boxes_list += [[xmin,ymin,xmax,ymax]]
                
    return boxes_list

def depth_map(data_dir):
    depth_dir = os.path.join(data_dir, 'Depth/')
    rgb_dir = os.path.join(data_dir, 'Images/')
    seg_dir = data_dir + 'Lane_Seg/Mask_RCNN_BDD100k_Gray/'
    annotation_dir = data_dir+'Annotations/'
    annotations = os.listdir(annotation_dir)
    annotations.sort()

    idss = 0

    depths = os.path.join(depth_dir, str(idss).zfill(4) + '.png')
    rgbs = os.path.join(rgb_dir, str(idss).zfill(4) + '.png')
    annotation_path = annotation_dir+annotations[idss]

    img = cv2.imread(depths)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.imread(rgbs)

    cv2.imshow("img_rgb", img_rgb)
    cv2.waitKey(0)

    # Seg dir
    seg_img_path = os.path.join(seg_dir, str(idss).zfill(4) + '-seg.png')
    seg_img = cv2.imread(seg_img_path,0)
    seg_height, seg_width = seg_img.shape

    list_colour = []

    # Unused
    # for seg in range(0, seg_width):
    #     if seg_img[299][seg] == 1 :
    #         list_colour.append(seg)

    # avg_seg = np.mean(list_colour)

    # max_value = max(list_colour)
    # min_value = min(list_colour)
    # lebar_jalan = max_value - min_value

    # cari titik minimum
    min_y = []
    min_x = []
    for y in range(0, seg_height):
        for x in range(0, seg_width):
            if seg_img[y][x] == 1 :
                min_x.append(x)
                min_y.append(y)

    # Remove the duplicate
    min_y = list(dict.fromkeys(min_y))
    min_x = list(dict.fromkeys(min_x))
    
    # Get the max and min for ltx rtx
    min_y_list = min(min_y)
    list_colour_point = []

    for x in range(0, seg_width):
        if seg_img[min_y_list+10][x] == 1 :
            list_colour_point.append(x)

    ltx_temp = min(list_colour_point)
    rtx_temp = max(list_colour_point)
    # END

    boxes_list = get_boxes_from_voc(annotation_path)

    avg_total = np.mean(img)

    bbox_store = []
    distance_store = []

    for i in range(len(boxes_list)):
        xmin = boxes_list[i][0]
        ymin = boxes_list[i][1]
        xmax = boxes_list[i][2]
        ymax = boxes_list[i][3]
        crop_img = img[ymin:ymax, xmin:xmax]
        avg = np.mean(crop_img)

        cp = avg
        ap = avg_total
        tp = 255

        RAT = ap/tp
        RCA = cp/ap
        RCT = cp/tp

        eRAT = math.exp(RAT)
        eRCA = math.exp(RCA)
        eRCT = math.exp(RCT)

        # distance = (pow(eRAT, 3) / pow(eRCA, 3)) * cp /  eRCT           # 1
        # distance = (pow(eRAT, 3) / pow(eRCT, 3)) * cp /  pow(eRCA, 2)`  # 2
        distance = pow(2,(eRAT/RCA)) * cp / pow(8,eRCT)                 # 3
        # distance = pow(15,(eRAT/eRCA)) * cp / pow(12,eRCT)              # 4

        ###############################
        # orde 3 polunomial
        distances = -20.67782 + 4.718018*distance - 0.1638691*pow(distance, 2) + 0.002051196*pow(distance, 3)

        # Save the bbox if it's fit the requirement
        if 10 <= distance <= 40: 
            w = xmax - xmin
            h = ymax - ymin
            print("w = ", w)
            print("h = ", h)

            if 40 <= w <= 175 and 30 <= h <= 150:
                bbox_store.append(boxes_list[i])
                distance_store.append(distance)
                print("distance = ", distance)

                crop_img_rgb = img_rgb[ymin:ymax, xmin:xmax]
                cv2.imshow("b", crop_img_rgb)
                cv2.waitKey(0)

        print("")
        ####################################################################################################

    ##############################################################################################################
    original_height, original_width, num_channels = img_rgb.shape
    original_center = abs(original_width/2)

    lty_store = []
    rty_store = []

    bird_width2 =  175
    bird_hight2 =  600

    for j in range(len(bbox_store)):
        xmin = bbox_store[j][0]
        ymin = bbox_store[j][1]
        xmax = bbox_store[j][2]
        ymax = bbox_store[j][3]
        jarak = distance_store[j]

        if 10 <= jarak <= 15:
            jarak_pix = 37
        elif 15 < jarak <= 20:
            jarak_pix = 33
        elif 20 < jarak <= 25:
            jarak_pix = 27
        elif 25 < jarak <= 30:
            jarak_pix = 22
        elif 30 < jarak <= 35:
            jarak_pix = 17
        elif 35 < jarak <= 40:
            jarak_pix = 12

        lty = ymax - jarak_pix
        rty = ymax - jarak_pix

        lty_store.append(lty)
        rty_store.append(rty)


    print("ltx_store = ", lty_store)
    print("rty_store = ", rty_store)
    print("")
    print("ltx_temp = ", ltx_temp)
    print("rtx_temp = ", rtx_temp)

    if rtx_temp - ltx_temp >= 300 :
        ltx = ltx_temp + 190
        rtx = rtx_temp + 20
    else :
        ltx = ltx_temp - 30
        rtx = rtx_temp + 30

    lty = np.mean(lty_store)
    rty = np.mean(rty_store)

    if math.isnan(lty) :
        weighted = 0.1 * min_y_list
        lty = min_y_list - weighted
        rty = min_y_list - weighted

    ldx = -600
    ldy = 384
    rdx = 1880
    rdy = 384
    src_ai = np.float32([[ltx, lty], [rtx, rty], [ldx, ldy], [rdx, rdy]])
    dst_ai = np.float32([[0, 0], [bird_width2-1, 0], [0, bird_hight2-1], [bird_width2-1, bird_hight2-1]])
    M_ai = cv2.getPerspectiveTransform(src_ai, dst_ai)
    warped_img_ai = cv2.warpPerspective(img_rgb, M_ai, (bird_width2, bird_hight2))

    with h5py.File(data_dir+'parameters.h5','w') as f:
        f.create_dataset('src', data = src_ai)
        f.create_dataset('bird_hight', data = bird_hight2)
        f.create_dataset('bird_width', data = bird_width2)
        f.create_dataset('bird_channels', data = 3)   

    print("src_ai = ", src_ai)

    #cv2.imshow("warped_ai", warped_img_ai)
    #cv2.waitKey(0)

    # Decision
    def quit_figure(event):
        if event.key == 'x':
            print("FALSE")
            cnt.append(1)


    plt.yticks([600,500,400,300,200,100,0], [0,10,20,30,40,50,60])
    plt.xticks([35,70,105, 140, 175],[3.5,7,11.5, 14, 17.5])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.imshow(cv2.cvtColor(warped_img_ai, cv2.COLOR_BGR2RGB)) # Show results
    plt.grid()
    cnt = []
    cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.savefig(data_dir + 'One_Bird_Depth.png')
    plt.show()
    plt.close()

    if len(cnt) >= 1 :
        lane_tag = False
    else :
        lane_tag = True

    print("done")
    return  lane_tag 
    ##############################################################################################################

if __name__ == '__main__':
    data_dir = "/media/ferdyan/LocalDiskE/Hasil/dataset/itri/5/"
    tag = depth_map(data_dir)
    print("tag = ", tag)
    cv2.destroyAllWindows()
