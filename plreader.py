import pytesseract
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pl_reader_defs as pl
pytesseract.pytesseract.tesseract_cmd = (
    r'C:\Program Files\Tesseract-OCR\tesseract'
)

def main(img_file_path):
    # read image from pdf_to_img

    img1 = pl.pdf_to_img(img_file_path)
    
    lefts1, rights1, center1, df3 = pl.get_areas(img1)

    dialated21 = pl.hori_areas(img1)

    right_unboxed1, left_unboxed1, center_unboxed1 = pl.unboxing(rights1,lefts1,center1)

    # tl1, tr1, tc1, ac1 = pl.vertical_groups(left_unboxed1, right_unboxed1, center1)
    cents12 = []
    lists1 = [[left_unboxed1,'left'], [right_unboxed1,'right']]#, [center,'center']]

    for lst in lists1:
        p1 = pl.clusters(lst[0],lst[1])
        cents12.append(p1)

    cents13 = sum(cents12, [])
    # cents13 = sum(cents11, [])

    outs1 = pl.vert_output(cents13, df3)

    outs12 = sum(outs1, [])

    m1 = int(np.median([int(el[3]) for el in outs1]))
    m2 = int(np.median([int(el[5]) for el in outs1]))
    m = (m1,m2)

    output1 = []
    for point in outs1:
        if point[1] == 'left':
            # print((point[2],m1), (point[4],m2))
            output1.append([point[2],m1, point[4],m2])

    indiv_boxes = pl.individual_boxes(dialated21, output1)

    img_test = img1.copy() # cv2.imread(img_file_path)

    outs3 = []
    for box in indiv_boxes:
        img_crop = img_test[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]
        out_string = pytesseract.image_to_string(img_crop)
        # clean = text_clean(out_string)
        outs3.append(out_string)

    p = pl.text_clean(outs3)
    a = int(len(p)/len(output1))
    chunks = [p[x:x+a] for x in range(0, len(p), a)]

    output_df = pd.DataFrame(chunks).T.rename(columns={0: "a", 1: "b", 2: "c"})
    print(output_df)
    return output_df