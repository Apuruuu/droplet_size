import cv2
import numpy as np

def Pretreatment(img,c_x, c_y):
    img[0:512, 200:300] = 255
    img[0:512, 612:712] = 255
    img[0:100 , 200:712] = 255
    img[412:512, 200:712] = 255
    _,img_binary =cv2.threshold(img[c_y-int(area / 2) :c_y+int(area / 2) , c_x-int(area / 2) :c_x+int(area / 2) ],25,255,cv2.THRESH_BINARY)# 截取（328,128） 到（584,384）并二值化
    img_binary = cv2.bitwise_not(img_binary, img_binary) # 反色
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # 指定一个5*5的卷积核
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_OPEN, kernel)#开操作
    img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, kernel)#闭操作
    channel_b, _, _ = cv2.split(img_binary)
    return channel_b

def get_size(img):
    row_width = []
    for num_row in range(len(img)):
        row_width.append(len(img[num_row][img[num_row]>=128])) # 求每行的宽度
    top, bottom = 0, 0
    for num_width in range(len(row_width)): # 求高度
        if bottom == 0 and row_width[-(num_width+1)] >0 : bottom = num_width
        if row_width[-(num_width+1)] >= 15: top = num_width

    max_value = row_width.index(np.max(row_width))
    left = np.argmax(img[max_value])
    right = len(img) - np.argmax(img[max_value][::-1])

    return left, right, top, bottom

def draw_img(img, left, right, top, bottom):
    img = cv2.merge([img, img, img])
    width = img.shape[0]
    height = img.shape[1]
    top = height - top
    bottom = height - bottom
    cv2.line(img, (int(width / 2) -15,int(height / 2)), (int(width / 2) +15,int(height / 2)), (0,255,0),thickness=1)
    cv2.line(img, (int(width / 2),int(height / 2)-15), (int(width / 2),int(height / 2)+15), (0,255,0),thickness=1)
    cv2.line(img, (left,int(height / 2)-15), (left,int(height / 2)+15), (255,255,0),thickness=1) # left
    cv2.line(img, (right,int(height / 2)-15), (right,int(height / 2)+15), (255,255,0),thickness=1) # right
    cv2.line(img, (int(width / 2)-15,top), (int(width / 2)+15,top), (255,255,0),thickness=1) # top
    cv2.line(img, (int(width / 2)-15,bottom), (int(width / 2)+15,bottom), (255,255,0),thickness=1) # left

    c_x = (left + right) / 2
    c_y = (top + bottom) / 2

    return img, c_x, c_y

def draw_frame(frame, c_x, c_y):
    cv2.line(frame, (c_x-15, c_y), (c_x+15, c_y), (0,255,0),thickness=1)
    cv2.line(frame, (c_x,c_y-15), (c_x,c_y+15), (0,255,0),thickness=1)
    cv2.rectangle(frame, (c_x-int(area / 2) , c_y-int(area / 2) ), (c_x+int(area / 2) , c_y+int(area / 2) ), (0, 0, 255), 2)
    
    return frame

if __name__ == '__main__':
    video_file = '20210727_144015_C001H001S0001.avi'
    cap = cv2.VideoCapture(video_file)
    c_x, c_y = 456 , 256
    mmpx = 0.0173
    area = 200 # 红框边长
    with open("%s.csv"%video_file, 'w') as f:
        while(True):
            ret, frame = cap.read()
            if not ret : break
            img =  frame.copy()
            img = Pretreatment(img, c_x, c_y)
            left, right, top, bottom = get_size(img)            
            frame= draw_frame(frame, c_x, c_y)
            img, _c_x, _c_y = draw_img(img, left, right, top, bottom)
            c_x = int(c_x - ((int(area / 2)  - _c_x) / 2))
            c_y = int(c_y - ((int(area / 2)  - _c_y) / 2))
            img = cv2.resize(img, (200,200))
            frame[312:512, 0:200] = img
            width = right - left
            height = top - bottom
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'W = %.3f mm'%(width*mmpx), (5, 180), font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, 'H = %.3f mm'%(height*mmpx), (5, 210), font, 0.5, (255, 255, 255), 1)
            

            cv2.imshow('s',frame)
            cv2.waitKey(10)

            f.write('%d, %d\n'%(int(width*mmpx), int(height*mmpx))) # 单位 微米