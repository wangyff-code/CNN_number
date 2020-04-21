import cv2
from run import my_CNN_class

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
my_cnn = my_CNN_class()

dec_size = 38

while True:
    ret, frame = cap.read()
    if ret == True:
        y,x,deep = frame.shape
        break

while True:
    ret, frame = cap.read()
    if ret == True:
        img = frame[y//2-dec_size:y//2+dec_size,x//2-dec_size:x//2+dec_size]
        number,img=my_cnn.dect_number(img)
        print(number)
        img = cv2.resize(img,(400,400))
        cv2.imshow('b',img)
        if number != None:
            frame = cv2.putText(frame, '{}'.format(number), (x//2-dec_size,y//2-dec_size), font, 1.2, (255, 255, 255), 2)
        else:
            frame = cv2.putText(frame, 'NO NUMBER', (x//2-dec_size-10,y//2-dec_size-10), font, 1.2, (255, 255, 255), 2)
        cv2.rectangle(frame, (x//2-dec_size,y//2-dec_size), (x//2+dec_size,y//2+dec_size), (0, 0, 255), 2, 4)
        cv2.imshow('a',frame)
        cv2.waitKey(100)
