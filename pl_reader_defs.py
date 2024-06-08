import pytesseract
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import fitz
import imutils
from collections import Counter
import re
from sklearn.cluster import DBSCAN
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract'
)

def pdf_to_img(file_name):
    # file_name = r"C:\Users\rgood\Downloads\income-statement-sample-1.pdf"
    # name = file_name.split('.')[0]
    # name = name.split('/')

    # open the file
    # pdf_file = fitz.open(file_name)
    pdf_file =fitz.open(filename=file_name) if type(file_name) is str else fitz.open(stream=file_name)

    for page in pdf_file:
        zoom_x = 3.0  # horizontal zoom
        zoom_y = 3.0  # vertical zoom
        mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension
        pix = page.get_pixmap(matrix=mat)
        # pix = page.get_pixmap(matrix=fitz.Identity, dpi=None,
        #                       colorspace=fitz.csRGB, clip=None, alpha=False, annots=True)
        img = np.array(Image.frombytes('RGB', [pix.width, pix.height], pix.samples))
    return img

def get_areas(img):
    image = img.copy()

    # compute the Scharr gradient magnitude representation of the images
    # in both the x and y direction using OpenCV 2.4
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(image, ddepth=ddepth, dx=1, dy=0, ksize=3)
    gradY = cv2.Sobel(image, ddepth=ddepth, dx=0, dy=1, ksize=3)
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.medianBlur(gradient, 5, 15)
    (_, thresh) = cv2.threshold(blurred, 100, 100, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (42, 6))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # h, w, p = image.shape
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 2))
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3,2))

    closed1 = cv2.erode(closed, None, iterations = 6)
    closed2 = cv2.dilate(closed1, None, iterations = 8)
    closed2 = cv2.cvtColor(closed2, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(closed2.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)
    # compute the rotated bounding box of the largest contour\]

    boxes = []
    left_upper = []
    left_lower = []
    right_upper = []
    right_lower = []
    center = []

    for c1 in c:
        rect = cv2.minAreaRect(c1)
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.int0(box)

        M = cv2.moments(c1)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        center.append([cX,cY])

        box_sorted = box[box[:, 0].argsort()]
        boxes.append(box_sorted.tolist())
        left_upper.append(tuple(box_sorted[0]))
        left_lower.append(tuple(box_sorted[1]))
        right_upper.append(tuple(box_sorted[2]))
        right_lower.append(tuple(box_sorted[3]))
        # draw a bounding box arounded the detected barcode and display the
        # image
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    # for point in right_upper:
    #     cv2.circle(image,point,5, (0,0,255), thickness = 5)
    # for point in left_upper:
    #     cv2.circle(image,point,5, (0,255,255),thickness = 5)
    # for point in right_lower:
    #     cv2.circle(image,point,5, (255,0,255),thickness = 5)
    # for point in left_lower:
    #     cv2.circle(image,point,5, (255,0,0),thickness = 5)
    # for point in center:
    #     cv2.circle(image,point,5, (255,255,0),thickness = 5)

    lefts = [left_upper,left_lower]
    rights = [right_upper, right_lower]
    points = [boxes,center, left_lower,left_upper,right_lower,right_upper]

    df1 = pd.DataFrame(points).T.rename(columns = {1:'center',0: 'all',2:'left_lower',3:'left_upper', 4:'right_lower', 5:'right_upper'})

    llx = [x[0] for x in df1['left_lower']]
    lux = [x[0] for x in df1['left_upper']]
    rlx = [x[0] for x in df1['right_lower']]
    rux = [x[0] for x in df1['right_upper']]
    cx = [x[0] for x in df1['center']]

    lly = [x[1] for x in df1['left_lower']]
    luy = [x[1] for x in df1['left_upper']]
    rly = [x[1] for x in df1['right_lower']]
    ruy = [x[1] for x in df1['right_upper']]
    cy = [x[1] for x in df1['center']]

    points = [llx,lux,rlx,rux,cx,lly,luy,rly,ruy,cy]
    df2 = pd.DataFrame(points).T.rename(columns = {0:'llx',1:'lux',2:'rlx',3:'rux',4:'cx',5:'lly',6:'luy',7:'rly',8:'ruy',9:'cy',})
    df2

    df = pd.concat([df1,df2], axis=1)

    return lefts, rights, center, df

def hori_areas(img):
    image2 = img.copy()
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.bitwise_not(image2)
    # test1 = np.any(image2, axis=1)
    # test = np.invert(test1).reshape(-1,1)

    h, w = image2.shape
    h
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2*w, 1))

    dilated = cv2.dilate(image2, kernel)
    dilated2 = np.where(dilated > 200, 255, 0)

    # plt.figure(figsize=(20,12))
    # plt.imshow(dilated2)

    return dilated2

def unboxing(rights, lefts, center):
    right_unboxed = sum(rights, [])
    left_unboxed = sum(lefts, [])
    center_unboxed = sum(center, [])

    return right_unboxed, left_unboxed, center_unboxed

# def vertical_groups(left_unboxed, right_unboxed, center):
#     tl=[]
#     tr=[]
#     tc=[]
#     ac = []
#     for i in left_unboxed:
#         tl.append((i[0],0))
#     for i in right_unboxed:
#         tr.append((i[0],0))
#     for i in center:
#         tc.append((i[0],0))
#     for i in center:
#         ac.append((i[1],0))
#     return tl, tr, tc, ac

def clusters(lst, side):
    X = np.array(lst)

    # model
    db = DBSCAN(eps=125, min_samples=5)
    db.fit_predict(X)
    labels = db.labels_

    # the number of clusters found by DBSCAN
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found by DBSCAN: {n_clusters}")

    center_clusters = []
    for label in set(labels):
        l = X[labels==label,:]
        ll = [x[0] for x in l]
        points_of_cluster_0 = X[labels==label,:]
        centroid_of_cluster_0 = np.mean(points_of_cluster_0, axis=0) 
        if label >= 0:
            center_clusters.append([centroid_of_cluster_0[0],ll,side])
            # print(label, centroid_of_cluster_0[0])
    
    return center_clusters

def vert_output(cents1,df):
    outs =[]
    for cent1 in cents1:
        if cent1[2] =='left':
            t1 = df['lux'].isin(cent1[1])
            t2 = df['llx'].isin(cent1[1])
            o = df[t1 | t2]

        if cent1[2] =='right':
            t1 = df['rux'].isin(cent1[1])
            t2 = df['rlx'].isin(cent1[1])
            o = df[t1 | t2]

        if cent1[2] =='center':
            t1 = df['cx'].isin(cent1[1])
            o = df[t1]
        out = [
            cent1[0],
            cent1[2],
            o['lux'].min(),
            o['luy'].min(),
            o['rlx'].max(),
            o['rly'].max(),
            o.shape]
        outs.append(out)
    return outs

# def sanity_check():
#     data_box = img.copy()
#     # output[0][1] : output[0][3], output[0][0] : output[0][2]
#     for i in output:
#         print(i)
#         data_box = cv2.rectangle(data_box, (i[0]-5,i[1]), (i[2]+5,i[3]), (0, 128, 128), 1)
#     data_box = cv2.cvtColor(data_box, cv2.COLOR_BGR2GRAY)

#     blended = (data_box.astype(float) + dilated2.astype(float)) / 2
#     plt.figure(figsize=(20,16))
#     return plt.imshow(blended)

def individual_boxes(dilated2, output):
    text_block=[]

    for dia in range(0,dilated2.shape[0]):
        if dilated2[dia][0] == 255:
            text_block.append(dia)


    vert_groups = []
    temp_box = []

    temp_box.append(text_block[0])

    for text in range(1,len(text_block)):
        if text_block[text] == text_block[text-1]+1:
            temp_box.append(text_block[text])
        else:
            if len(temp_box) < 5:
                temp_box = []
                continue
            else:
                vert_groups.append((min(temp_box)-3,max(temp_box)+3))
                temp_box = []


    indiv_boxes=[]

    for op in output:
        for j in vert_groups:
            if j[0]< output[0][1]-1:
                continue
            indiv_boxes.append([j[0]-5,j[1]+5,op[0]-5,op[2]+5])
        # print(i)
    indiv_boxes.sort(key=lambda x: x[3])
    return indiv_boxes

def box_to_text(img_file_path,indiv_boxes):
    img_test = cv2.imread(img_file_path)
    outs = []
    for box in indiv_boxes:
        img_crop = img_test[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]
        out_string = pytesseract.image_to_string(img_crop)
        # clean = text_clean(out_string)
        outs.append(out_string)
    return outs

def str_to_int(strObj):
    strObj_l = [*strObj]
    if len(strObj_l) == 0:
        return strObj
    elif strObj_l[0] =='(' and strObj_l[1].isdigit():
        # return strObj
        # if :
        strObj = re.sub("\(","-",strObj)
        strObj = re.sub("\)","",strObj)
        return strObj
    else:
        return strObj

def text_clean(text):
    # text = re.sub("\n","",text)
    # text = filter(str.isal)
    # text = "".join(c for c in text if c.isalnum())
    # text = re.sub(r"[\.]", "", text)
    text = [re.sub("\n","",line) for line in text]
    text = [re.sub("\x0c","",line) for line in text]
    # text = [re.sub("\(\d+\)",r"\1",line) for line in text]
    # text = [re.sub("\(","-",line) for line in text]
    # text = [re.sub("\)","",line) for line in text]
    # text = [re.sub("[\),]","",line) for line in text]
    text = [re.sub("[^A-Za-z0-9\- ()]","",line) for line in text]
    text = [str_to_int(c) for c in text]
    # text = [c for c in text if c.isalnum()]
    # text = [re.sub(r'(\s)', '_', file) for file in text]
    return text
