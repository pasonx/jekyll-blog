---
layout: post
title:  "image augmentation processing"
date:   2017-07-04 22:59:13 +0000
categories: 
---
The code written in Python is used for object detection. We use deep learning method to realize the detection job. The first step is we need to guarantee that we have enough training images set for training the neural networks. I get the images data from American medical disease institute.
What the code below do is to augmente the image in different dimension and angle so that we can gain more data for tarining

{% highlight python %}
import xlrd
import glob
from PIL import Image
import matplotlib.pyplot as plt
import scipy as sp

import os
import numpy as np
import scipy.misc as misc
import dicom
import cv2

def get_data(file = 'list_size.xls', patient = 0, diam = 4, x = 5, y = 6, slice_num = 7, by_name=u'list3.2'):
    #for more detail of how xlrd work, please just google it
    #read data from Excel file
    data  = xlrd.open_workbook(file)
    table = data.sheet_by_name(by_name)
    #get the corresponding column
    col_case = table.col_values(patient)
    col_diam = table.col_values(diam)
    col_x = table.col_values(x)
    col_y = table.col_values(y)
    col_slice_value = table.col_values(slice_num)

    case = []
    case.append(int(col_case[1]))
    case.append(col_diam[1])
    case.append(int(col_x[1]))
    case.append(int(col_y[1]))
    case.append(int(col_slice_value[1]))

    for item in case:
        print item
    return case

def image_resize(img, gts):
    rand_coff_array = [0.5, 0.65, 0.85, 1.2, 1.3, 1.5]
    n = len(rand_coff_array)
    index = np.random.randint(0,n-1)
    resize_coff = rand_coff_array[index]
    img =  misc.imresize(img, resize_coff)
    gts =  misc.imresize(gts, resize_coff)
    return img, gts


def load_image(target, diam, x, y):
    for image in glob.glob('./*.dcm'):
        ds = dicom.read_file(image)
        if ds.InstanceNumber == target:
            #make sure the value of ds.pixel_array >= 0
            ds_array = (ds.pixel_array + 1024)
            ds_array = ds_array / (ds_array.max()*1.0)
            #draw a red rectangle to surround the nodule
            cv2.rectangle(ds_array, (int(x-diam/2)-8, int(y-diam/2)-8), (int(x+diam/2)+8, int(y+diam/2)+8), (0, 0,255), 2)
            #though the processing above, we can show the image as the following way
            #cv2.imshow("laod_img", ds_array)
            #cv2.waitKey()
            print "Data type", ds_array.dtype
            #note that when we need write the image to disk, we should alter the data type of every pixel
            #Just like this
            #ds_array = (ds_array * 255).astype(np.uint8)
            #cv2.imwrite("./mark.jpg", ds_array)
            return ds

def window(ds,L,H):
    I = ds.pixel_array
    low = (L - ds.RescaleIntercept) / ds.RescaleSlope
    high = (H - ds.RescaleIntercept) / ds.RescaleSlope
    #normlize
    I = I * (I > low) + (low * ( I <= low))
    I = I * (I < high) + (high * (I >= high))
    return (((I - low) / ((high - low) + 0.0)) * 255).astype(np.uint8)

def CT2RGB(ds):
    #get there different scope gray_img and combine them into RGB
    AN = window(ds,-950, 240)#normal image
    AH = window(ds,-950, -150)# highimage
    AL = window(ds,-160, 240)#low image
    # mask = img_segmentation(dicom_file)
    # mask = mask.astype(np.uint8)
    # AN = mask * AN
    # AH = mask * AH
    # AL = mask * AL
    img = np.dstack([AL,AN,AH])
    #cv2.imshow('test',img)
    #cv2.waitKey()
    return img

def image_crop(img, gts):
    (m,n) = img.shape[:-1]
    (row, col) = np.where(gts)
    re = max(max(col) - min(col), max(row) - min(row))
    a = 6
    b = 2
    cpara1 = [a, a, b, b]
    cpara2 = [b, b, a, a]
    cpara3 = [a, b, b, a]
    cpara4 = [b, a, a, b]
    cpara5 = [a, a, a, a]

    cpara = [cpara1, cpara2, cpara3, cpara4, cpara5]
    list_length = len(cpara)
    index = np.random.randint(0, list_length - 1)
    crop_coff = cpara[index]
    sxl = crop_coff[0]
    syl = crop_coff[1]
    sxr = crop_coff[2]
    syr = crop_coff[3]
    x0 = min(row) - sxl * re
    y0 = min(col) - syl * re

    xz = x0 + (sxl + 1 + sxr) * re
    yz = y0 + (syl + 1 + syr) * re

    print "debug0", sxl, syl, sxr, syr, x0, y0, re
    if x0 < 0:
        x0 = 0

    if y0 < 0:
        y0 = 0

    if xz > n:
        xz = n

    if yz > m:
        yz = m

    w = xz - x0
    print "debug", x0, y0, xz, yz
    print w
    h = yz - y0
    print h
    imgc = img[y0:y0+h, x0:x0+w]
    gtsc = gts[y0:y0+h, x0:x0+w]
    cv2.imshow("Flipped Horizontally img", imgc)
    cv2.imshow("Flipped Horizontally gts", gtsc)
    cv2.waitKey()
    return imgc, gtsc






def image_rotate(imgs, gts):
    rand_coff_array = [-40, -30, -15, 10, 35, 60]
    n = len(rand_coff_array)
    index = np.random.randint(0, n - 1)
    rotate_coff = rand_coff_array[index]
  #  cv2.imshow("test", misc.imrotate(imgs, rotate_coff))
  #  cv2.imshow("abc", misc.imrotate(gts, rotate_coff))
  #  cv2.waitKey()
    return misc.imrotate(imgs, rotate_coff), misc.imrotate(gts, rotate_coff)

def image_flip(imgr,gts):
    imgr = cv2.flip(imgr, 1)
    gts = cv2.flip(gts,1)
    #cv2.imshow("Flipped Horizontally", imgr)
    #cv2.waitKey()
    return imgr,gts

def generate_zeros_picture(img_shape,diam, x, y):
    img = np.zeros(img_shape[:-1],dtype=np.uint8 )
    xl = int(x-diam/2)-8
    xr = int(x+diam/2)+8
    yt = int(y-diam/2)-8
    yb = int(y+diam/2)+8
    #mark the nodoule area
    img[yt:yb,xl:xr] = 255
    #print xl,yt,xr,yb
    #cv2.imwrite("./mark1.jpg", img)
    cv2.imshow("test", img)
    cv2.waitKey()
    return img



def get_the_dicom_image(rootDir, dcm_file_list, target):
	list = os.listdir(rootDir)
	for item in list:
		path = os.path.join(rootDir, item)
		if os.path.isdir(path):
			get_the_dicom_image(path, dcm_file_list)
		else:
			if '.dcm' in item:
				ds = dicom.read_file(path)
				dcm_file_list.append(ds)
				print item
	        	if ds.InstanceNumber == target:
	            		ds_array = (ds.pixel_array + 1024)
	            		ds_array = ds_array/(ds_array.max()*1.0)
                        target_ds = ds
	return dcm_file_list, target_ds

#Create a 3-dimensional cubic V
#x 512 * y 512 * z 133
def create_3_dim(dcm_file_list):
    # x 512 * y 512 * z 133
    img = np.zeros((dcm_file_list[0].Rows, dcm_file_list[0].Columns, len(dcm_file_list)))
    print 'wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww',img.shape
    pixel_space = dcm_file_list[0].PixelSpacing[0]
    slice_thichness = dcm_file_list[0].SliceThickness
    rate = int(slice_thichness / pixel_space)

    for j in range(len(dcm_file_list)):
        #print dcm_file_list[j].InstanceNumber
        ds_array = dcm_file_list[j].pixel_array + 1024
        ds_array = ds_array / (ds_array.max() * 1.0)
        img[:, :, dcm_file_list[j].InstanceNumber-1] = ds_array


    cv2.imshow("NPU",  img[:, :, 112])
    cv2.waitKey()

    m = img.shape[0]
    n = img.shape[1]
    l = img.shape[2]

    print m,n,l
    # cv2.imshow("1", img[:,200,:])
    # cv2.waitKey()
    #512 * 133 * 512
    x_view_image = np.zeros((m,l,n))
    # 512 * 133 * 512
    y_view_image = np.zeros((n,l,m))

    print "==========="
    # cv2.imshow("a", img[:,256,:])
    # cv2.waitKey()
    #
    for i in range(n):
        #Generate
        imgx = img[:,i,:]
        x_view_image[:,:,i] = imgx
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~',imgx.shape
    # cv2.imshow("++++++++++++++", x_view_image[:,:,200])
    # cv2.waitKey()
    print "==========="

    for j in range(m):
        imgy = img[j,:,:]
        y_view_image[:,:, j] = imgy

    print "==========="





    cv2.imshow("1", x_view_image[:,:, 200])
    cv2.imshow("2", y_view_image[:,:, 200])
    cv2.waitKey()
    #

    print "Hello"
    return x_view_image, y_view_image,rate

    #print np.size(img)


def test(x_view,y_view,dcm_file_list):
    pixel_space = dcm_file_list[0].PixelSpacing[0]
    slice_thichness = dcm_file_list[0].SliceThickness
    image_x_ = []
    image_y = []
    print x_view.shape
    for l in range(x_view.shape[0]):
        image_x = x_view[l,:, :]
        image_size = x_view.shape
        a = cv2.resize(image_x, (len(dcm_file_list) * int((slice_thichness / pixel_space)), image_size[1]))
        #print "MIUI"+str(a.shape)

        image_x_.append(a)
    for image in image_x_:
        pass
        #print 'shape',image.shape
    image_x_ = np.array(image_x_)
    #print y_view.shape
    # for l in range(y_view.shape[0]):
    #     image_y = y_view[l,:, :]
    #     image_size = y_view.shape
    #     image_y.append(cv2.resize(image_y, (len(dcm_file_list) * int((slice_thichness / pixel_space)), image_size[0])))
    # image_y = np.array(image_y)

    print '/////////////////////////////////////////////',image_x_.shape
    # cv2.imshow("a", image_x_[85,:,:])
    # #cv2.imshow("b", image_y[80,:,:])
    # cv2.waitKey()

def generate_gts(dcm_file_list, diam, slice_img, target):
    pixel_space = dcm_file_list[0].PixelSpacing[0]
    slice_thichness = dcm_file_list[0].SliceThickness


    g = np.zeros((dcm_file_list[0].Rows, dcm_file_list[0].Columns, len(dcm_file_list)),dtype=np.uint8)
    num_layer = np.round(diam / 2.5)
    print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", dcm_file_list[0].InstanceNumber
    for i in range(int(-num_layer/2), int(num_layer / 2)):
        g[:,:,target+i] = slice_img

    cv2.imshow("---------------------------------", slice_img)
    cv2.waitKey()

    cv2.imshow("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",g[:,:,target])
    cv2.waitKey()

def projection(x_view, y_view, rate):
    m, l, n = x_view.shape[0], x_view.shape[1], x_view.shape[2]
    x = []
    y = []
    for i in range(1, n - rate, rate):
        temp = np.zeros((m, l))

        index_end = i + rate - 1
        if index_end > 512:
            index_end = 512

        for j in range(i, index_end):
            temp = temp + x_view[:, :, j]
        #print "*********************************"+str((temp / rate).shape)
        x.append(temp / rate)
    x=np.array(x)

    for i in range(1, m - rate, rate):
        temp = np.zeros((n, l))
        index_end = i + rate - 1
        if index_end > 512:
            index_end = 512
        for j in range(i, index_end):
            temp = temp + y_view[:, :, j]
        y.append(temp / rate)
    y = np.array(y)

    # cv2.imshow("a", x[75,:,:])
    # cv2.waitKey()

    return x, y






def main():
    #Read the corresponding information from the given XML file
    #Note that case is a list object used to contain the appending element / item
    case = get_data()
    #The load_image function return the InstanceNumber image
    dcm_file_list = []
    filePath = '/home/accurad/test'
    dcm_file_list, ds = get_the_dicom_image(filePath, dcm_file_list, case[4])
    #deplicated method
    #ds = load_image(case[4], case[1], case[2], case[3])
    #adjust the window of image
    #get three different scope gray_img and combine them into RGB
    img = CT2RGB(ds)
    #generate gts pic
    gts = generate_zeros_picture(img.shape, case[1], case[2], case[3])
    # # img, gts = image_resize(img, gts)
    # # cv2.imshow("test", img)
    # # cv2.waitKey()
    # # cv2.imshow("wps", gts)
    # # cv2.waitKey()
    # # img, gts = image_crop(img, gts)
    # # cv2.imshow("in main", gts)
    # # cv2.waitKey()
    # imgc , gtsc = image_crop(img, gts)
    # cv2.imwrite("./ttttttttttttt.jpg", imgc)
    # cv2.imwrite("./sssssssssssss.jpg", gtsc)
    # img02, gts02 = image_resize(img, gts)
    # img23, gts23 = image_rotate(img02, gts02)
    # img24, gts24 = image_flip(img02, gts02)
    # img234, gts234 = image_flip(img23, gts23)
    # img03, gts03 = image_rotate(img, gts)
    # img34, gts34 = image_flip(img03, gts03)
    # img04, gts04 = image_flip(img, gts)
    #
    # img_set = [imgc,img02,img23,img24,img234,img03,img34,img04]
    # gts_set = [gtsc,gts02,gts23,gts24,gts234,gts03,gts34,gts04]
    #
    # print "length", len(img_set)
    #
    # file_object = open('./thefile.txt', 'w')
    # for i in range(len(img_set)):
    #     cv2.imwrite("./gts" + str(i) + ".jpg", gts_set[i])
    #     wi = []
    #     hi = []
    #     for h in range(gts_set[i].shape[0]):
    #         for w in range(gts_set[i].shape[1]):
    #             test = gts_set[i]
    #             if test[h][w] == 255:
    #                 #print h, w
    #                 wi.append(w)
    #                 hi.append(h)
    #
    #     text_item = "img" + str(i) + ".jpg" + " " + str(min(hi)) + " " + str(min(wi)) + " " + str(max(hi)) + " " + str(
    #         max(wi)) + "\n"
    #     file_object.writelines(text_item)
    #     cv2.rectangle(img_set[i], (int(min(wi)), int(min(hi))),(int(max(wi)), int(max(hi))), (0, 0, 255), 2)
    #     cv2.imshow("laod_img", img_set[i])
    #     cv2.waitKey()
    #
    #     #Note that this part of code is using OpenCV to draw the












   #  for item in img_set:
   #    #  item = (item * 255).astype(np.uint8)
   #      cv2.imwrite("./img" + str(i) + ".jpg", item)
   #      i += 1
   #  i = 0
   #  for item in gts_set:
   # #     item = (item * 255).astype(np.uint8)
   #      cv2.imwrite("./gts" + str(i) + ".jpg", item)
   #      i += 1
   #
   #
   #  i = 0
   #  file_object = open('./thefile.txt', 'w')
   #  for item in gts_set:
   #      #print "w, h", item.shape[0], item.shape[1]
   #      wi = []
   #      hi = []
   #      for h in range(item.shape[0]):
   #          for w in range(item.shape[1])  :
   #              if item[h,w] == 255:
   #                  print h, w
   #                  wi.append(w)
   #                  hi.append(h)
   #      print "========================================================================"
   #      print "\n"
   #      print "\n"
   #      print "\n"
   #      print "\n"
   #      print "\n"
   #      print "\n"
   #      print i
   #      print hi
   #
   #      print wi
   #      img_set(i)
   #      text_item =  "img" + str(i) + ".jpg" + " " +  str(min(hi)) + " " + str(min(wi)) +  " " + str(max(hi)) + " " + str(max(wi)) + "\n"
   #      i = i + 1
   #      file_object.writelines(text_item)


    x_view, y_view, rate = create_3_dim(dcm_file_list)
    x, y = projection(x_view, y_view, rate)
    test(x, y, dcm_file_list)
    generate_gts(dcm_file_list,case[1],gts,case[4])
    print "finished!"


if __name__=="__main__":
        main()

{% endhighlight %}

