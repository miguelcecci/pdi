import cv2
import numpy as np
from matplotlib import pyplot as plt

gray = cv2.imread('../marked-tests/cnh-test-3.jpg')
final_img = gray

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.3.0_3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

#rotate
# rows, cols, _ = gray.shape

# filtro 2d
# kernel = np.ones((5,5), np.float32)/25
# gray = cv2.filter2D(gray, -1, kernel)

#blur

# M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
# gray = cv2.warpAffine(gray,M,(cols,rows))
#equalization
# gray = cv2.equalizeHist(gray)
# binarize
gray = cv2.blur(gray,(15,15))
# ret,gray = cv2.threshold(gray ,150,255,cv2.THRESH_BINARY)
# gray = cv2.adaptiveThreshold( gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,110,2)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    #desenha linha na metade da im
    # cv2.line(gray, (int(gray.shape[1]/2), 0), (int(gray.shape[1]/2), gray.shape[0]), (255, 0, 0), 1, 1)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    dsize = 5
    area_units = 0
    y_div = int(gray.shape[0]/dsize)
    x_div = int(gray.shape[1]/dsize)
    #segmentando a imagem
    img_segs = np.zeros((x_div, y_div), dtype=object)
    for i in range(0, x_div):
        for j in range(0, y_div):
            cropped = gray[j*dsize:(j+1)*dsize, i*dsize:(i+1)*dsize]
            avg_color = [np.floor(cropped[:, :, w].mean()) for w in range(cropped.shape[-1])]

            if avg_color[2] <= 90:
                cv2.rectangle(final_img,(int(i*dsize),int(j*dsize)),(int((i+1)*dsize),int((j+1)*dsize)),(255,0,0),1)
                area_units += 1
            else:
                cv2.rectangle(final_img,(int(i*dsize),int(j*dsize)),(int((i+1)*dsize),int((j+1)*dsize)),(0,0,0),1)
            img_segs[i][j] = avg_color

    print(area_units)



    for (x,y,w,h) in faces:
        # cv2.rectangle(gray,(int(x-w/3),int(y-h/2)),(int(x+w+w/3),int(y+h+h/2)),(0,0,0),1)
        roi_gray = gray[y:y+h, x:x+w]
    cv2.imshow('frame', gray)
    cv2.imshow('frame2', final_img)
