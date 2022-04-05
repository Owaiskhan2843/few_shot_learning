
def prep_img(img):
    import cv2
    import numpy as np
    from PIL import Image
    import io
    import string
    import time
    import os
    import math
    size=100
    hei=[]
    wei=[]
    c=0

    # # img=cv2.imread(path)
    # img = Image.open(io.BytesIO(img))
    
    starter = img.find(',')
    img = img[starter+1:]
    img = Image.open(io.BytesIO(base64.decodebytes(bytes(img, "utf-8")))).convert('RGB')
    # img = img.resize((256, 256))
    # img = Image.open(io.BytesIO(img))
    img = np.array(img, dtype='uint8')
    img = cv2.resize(img, (size, size), interpolation = cv2.INTER_AREA)
    oimg=img.copy()
    kernel=np.ones((3,3),np.uint8)
    cimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #80 30
    clahe = cv2.createCLAHE(clipLimit = 5) 
    final_img = clahe.apply(cimg) + 30
    # 80 33 for clahe size 80
    #80 42 for 100
    circles = cv2.HoughCircles(final_img,cv2.HOUGH_GRADIENT,1,20,param1=80,param2=42,minRadius=0,maxRadius=0)
    if circles is not None:
        c+=1
        detect_circles = np.uint16(np.around(circles)) 
        max_a=0
        for i in detect_circles[0,:]:
            # draw the outer circle
            ar=int(i[2]*i[2]*math.pi)
            if ar>max_a:
                x=i[0]
                y=i[1]
                r=i[2]
                max_a=ar
        x=int(x)
        y=int(x)
        r=int(x)
        height,width=cimg.shape
        mask = np.zeros((height,width), np.uint8)
        cv2.circle(mask,(x,y),r,(255,255,255),-1)
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

        # Find Contour
        cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

        #print len(contours)
        x,y,w,h = cv2.boundingRect(cnt[0])

        # Crop masked_data
        crop = masked_data[y:y+h,x:x+w]
        
        return crop
        
       