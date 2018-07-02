import numpy as np
from scipy.ndimage.filters import convolve
from numpy import uint8
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

def realtime_graph(x, y):
    line, = plt.plot(x, y,label="S Curve") # (x,y)のプロット
    line.set_ydata(y)   # y値を更新
    plt.title("S Curve")  # グラフタイトル
    plt.xlabel("x")     # x軸ラベル
    plt.ylabel("y")     # y軸ラベル
    plt.legend()        # 凡例表示
    plt.grid()          # グリッド表示
    plt.xlim([0,255])    # x軸範囲
    plt.ylim([0,255])    # y軸範囲
    plt.draw()          # グラフの描画
    plt.pause(0.01)     # 更新時間間隔
    plt.clf()

def myfunc(i):
    pass # do nothing

def S_curve(x):
    y = (np.sin(np.pi * (x/255 - s/255)) + 1)/2 * 255
    return y

cv2.namedWindow('title') # create win with win name

cv2.createTrackbar('red', # name of value
                   'title', # win name
                   100, # init
                   100, # max
                   myfunc) # callback func
cv2.createTrackbar('green', # name of value
                   'title', # win name
                   100, # init
                   100, # max
                   myfunc) # callback func
cv2.createTrackbar('blue', # name of value
                   'title', # win name
                   100, # init
                   100, # max
                   myfunc) # callback func
cv2.createTrackbar('Laplacian filter', # name of value
                   'title', # win name
                   0, # init
                   1, # max
                   myfunc) # callback func
cv2.createTrackbar('s_curve',
                   'title',
                   127,
                   255,
                   myfunc)
cv2.createTrackbar('S Curve:on/off',
                   'title',
                   0,
                   1,
                   myfunc)





cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


while(True):

    ret, frame = cap.read()
    if not ret: continue


    gamma = cv2.getTrackbarPos('gamma',  # get the value
    									   'title')  # of the win
    r = cv2.getTrackbarPos('red',  # get the value
    									   'title')  # of the win
    g = cv2.getTrackbarPos('green',  # get the value
    									   'title')  # of the win
    b = cv2.getTrackbarPos('blue',  # get the value
    									   'title')  # of the win
    switch = cv2.getTrackbarPos('Laplacian filter',  # get the value
    									   'title')  # of the win
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    kernel = np.array([[0, 1, 0],
                       [1,-4, 1],
                       [0, 1, 0]])

    ## do something by using v

    R = frame[:,:,2]
    G = frame[:,:,1]
    B = frame[:,:,0]
    R = R/255
    G = G/255
    B = B/255
    R = R*(r/100)
    G = G*(g/100)
    B = B*(b/100)
    frame[:,:,2] = R*255
    frame[:,:,1] = G*255
    frame[:,:,0] = B*255

    #S Curve
    s = cv2.getTrackbarPos('s_curve','title')
    x = np.linspace(0, 255, 100)
    y = S_curve(x)
    realtime_graph(x,y)

    #S Curve on/off
    sw = cv2.getTrackbarPos('S Curve:on/off','title')
    if sw == 1:
        frame = S_curve(frame)
    cv2.imshow('title', frame)  # show in the win

    if switch == 1:
        lframe = convolve(gray,kernel)
        frame = lframe
        frame = frame.astype(np.uint8)

    else :
        frame[:,:,2] = R*255
        frame[:,:,1] = G*255
        frame[:,:,0] = B*255



    cv2.imshow('title', frame)  # show in the win

    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break



cap.release()
cv2.destroyAllWindows()
