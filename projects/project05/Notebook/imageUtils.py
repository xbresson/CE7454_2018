import cv2
import numpy as np
import matplotlib.pyplot as plt

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def point_to_stroke(point):

    point = normalise_point(point)
    num_point = len(point)
    #stroke =[dx,dy,dt]

    #--------
    stroke    = np.zeros((num_point,3),np.float32)
    stroke[0] = [0,0,1]
    stroke[1:] = point[1:]- point[:-1]

    return stroke

def point_to_image(point, H, W, border=0.05):

    point = normalise_point(point)
    time  = point[:,2].astype(np.int32)
    norm_point = point[:,:2]
    norm_point = norm_point[:,:2]* max(W,H)*(1-2*border)
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

    #--------
    image = np.zeros((H,W,3),np.uint8)

    T = time.max()+1
    for t in range(T):
        p = norm_point[time==t]
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i]
            x1,y1 = p[i+1]
            cv2.line(image,(x0,y0),(x1,y1),(255,255,255),1,cv2.LINE_AA)

        x,y = p.T
        image[y,x]=(255,255,255)
    return image

def normalise_point(point):

    x_max = point[:,0].max()
    x_min = point[:,0].min()
    y_max = point[:,1].max()
    y_min = point[:,1].min()
    w = x_max-x_min
    h = y_max-y_min
    s = max(w,h)

    point[:,:2] = (point[:,:2]-[x_min,y_min])/s
    point[:,:2] = (point[:,:2]-[w/s*0.5,h/s*0.5])

    return point

def null_image_augment(point,label,index):
    cache = Struct(point = point.copy(), label = label, index=index)
    image = point_to_image(point, 64, 64)
    return image, label, cache

def null_stroke_augment(point,label,index):
    cache = Struct(point = point.copy(), label = label, index=index)
    stroke = point_to_stroke(point)
    return stroke, label, cache

def draw_point_to_overlay(point):
    H, W   = 256, 256
    border = 0.05

    point = normalise_point(point)
    time  = point[:,2].astype(np.int32)
    norm_point = point[:,:2]
    norm_point = norm_point[:,:2]* max(W,H)*(1-2*border)
    norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)

    #--------
    overlay = np.full((H,W,3),255,np.uint8)


    T = time.max()+1
    colors = plt.cm.jet(np.arange(0,T)/(T-1))
    colors = (colors[:,:3]*255).astype(np.uint8)

    for t in range(T):
        color = colors[t]
        color = [int(color[2]),int(color[1]),int(color[0])]

        p = norm_point[time==t]
        N = len(p)
        for i in range(N-1):
            x0,y0 = p[i  ]
            x1,y1 = p[i+1]
            cv2.line(overlay,(x0,y0),(x1,y1),color,2,cv2.LINE_AA)

        for i in range(N):
            x,y = p[i]
            #cv2.circle(overlay, (x,y), 4, [255,255,255], -1, cv2.LINE_AA)
            cv2.circle(overlay, (x,y), 4, [0,0,0], -1, cv2.LINE_AA)
            cv2.circle(overlay, (x,y), 3, color, -1, cv2.LINE_AA)



        # x,y = p.T
        # overlay[y,x]=(255,255,255)
    return overlay