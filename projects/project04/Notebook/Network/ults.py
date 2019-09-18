import cv2
import numpy as np
import os
import math
import glob
from skimage.draw import polygon
import dlib
os.chdir('/media/imi-yujun/579e63e9-5852-43a7-9323-8e51241f5d3a/yujun/Course_porject_fac e')

mu_shape = np.load("data/mu_shape_.npy")
mu_exp = np.load("data/mu_exp_.npy")
b_exp = np.load("data/b_exp_.npy")
b_shape = np.load("data/b_shape_.npy")
b_tex = np.load("data/b_tex_.npy")
mu_tex = np.load("data/mu_tex_.npy")
key_index = np.loadtxt('data/keyindex.txt',dtype=np.int).tolist()
def eulerAnglesToRotationMatrix( theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def CalculateDense(shape_,exp_):
    dmm = mu_exp + mu_shape + Blend(shape_, b_shape) +Blend(exp_, b_exp)
    return dmm

def Blend(weight, matrix):
    tmp = np.zeros(np.shape(matrix[0, :]))
    for i in range(len(weight)):
        tmp += weight[i] * matrix[i, :]
    return tmp


def RT():
    R=[ [0.997069291942560, - 0.00510382095553161,0.0763333352920970],[0.00470009564452208,0.999974006378591,0.00546769312252528],[-0.0763592572390157, - 0.00509289493349726,0.997067362947510]]
    R=np.reshape(np.array(R),(3,3))
    T=[ 125.062413828980,	77.8392149438461,	-107.601797147489]
    T=np.reshape(np.array(T),(1,3))

    path_ ="D:\HDface_Data0918/"
    #tmp = np.zeros((9,68,3))
    index_=0
    for i in range(0,22):
        points = np.loadtxt(path_+ "3dlmk/"+str(i)+".txt",delimiter=',')*1000
        points_  =np.matmul(points,R)+T;
        #tmp[index_,:] = points_
        #index_+=1
        np.savetxt(path_+'CMOS/3dlmk/'+str(i)+'.txt',points_,delimiter=",")


def FaceIndex(path = "data/resize_uv.obj"):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    face_points =[]
    for i in range(len(content)):
        tmp = content[i].split(" ")
        if(tmp[0]=="f"):
            face_points.append([ tmp[1].split("/")[0] ,tmp[2].split("/")[0],tmp[3].split("/")[0] ])
    face_points = np.array(face_points,dtype=np.int)-1
    return face_points

def flip():
    file_list_ = glob.glob("D:\HDface_Data0918\calibration/0\*.jpg")
    for i in range(len(file_list_)):
        img = cv2.imread(file_list_[i])
        img = np.flip(img,axis=1)
        cv2.imwrite(file_list_[i],img)

def PhotoChange():
    path= "D:\HDface_Data0918\calibration/1/"
    for i in range(1,10):
        img =cv2.imread(path+"cali_"+str(i)+'.jpg')[218:2848-218,:]
        img = cv2.resize(img,(1920,1080))
        cv2.imwrite(path+ str(i)+".jpg",img)

def TriScanLine():
    #Generate a map as 1024x1024
    #the value means index of face
    uv = np.loadtxt("data/uv_point.txt",delimiter=',')[:,:2]*1024
    face_index = FaceIndex()
    TriIndex= np.zeros((1024,1024),dtype=np.uint16)
    for i in range(len(face_index)):
        index_ = [face_index[i,0],face_index[i,1],face_index[i,2]]
        r = uv[index_,1]
        c =  uv[index_,0]
        rr, cc = polygon(r, c)
        TriIndex[1024-rr, cc] = i+1
    np.save("data/TriMap.npy",TriIndex)

def Get3Dmm(path):
    with open(path) as f:
        dmm_para = f.readlines()
    dmm_para = [x.strip() for x in dmm_para]
    shape_ = np.array(dmm_para[0].split(), dtype=np.float)
    exp_ = np.array(dmm_para[1].split(), dtype=np.float)
    tmp = np.array(dmm_para[2].split(), dtype=np.float)
    eular_   = tmp[:3]
    translate_ = tmp[3:5]
    scale_ = tmp[5]
    return shape_,exp_,eular_,translate_,scale_

def CMOSLmk():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('E:\Library\shape_predictor_68_face_landmarks.dat')
    save_path ="D:\HDface_Data0918\CMOS/2dlmk"
    img_path = "D:\HDface_Data0918\CMOS/"
    for i in range(0,22):
        img = cv2.imread(img_path+"/"+str(i)+'.jpg', cv2.COLOR_BGR2GRAY)
        pixel_pos = []
        try:
            dets = detector(img, 1)
            shape = predictor(img, dets[0])
            for j in range(68):  # 60 and 64 is not need
                pixel_pos.append([shape.part(j).x, shape.part(j).y])
        except:
            continue
        np.savetxt(save_path+"/"+str(i)+'.txt',np.array(pixel_pos,dtype= np.uint),fmt= "%d",delimiter=',')

def VertexColor(tex_):
    dmm_tex =  Blend(tex_, b_tex)+mu_tex
    return dmm_tex

def Cap(img):
    img[img > 255] = 255
    img[img < 0] = 0
    return img


def RidgeFit2D(X, Y):
    A = np.zeros((2 * len(X), 3))
    B = np.zeros((2 * len(Y)))
    for i in range((len(X))):
        A[2 * i, :] = np.array([X[i, 0], 1, 0])
        A[2 * i + 1, :] = np.array([X[i, 1], 0, 1])
        B[2 * i] = Y[i, 0]
        B[2 * i + 1] = Y[i, 1]
    ATA = np.linalg.inv((np.matmul(A.transpose(), A)))
    ATB = np.matmul(A.transpose(), B)
    ans = np.matmul(ATA, ATB)
    return ans[0],ans[1:]

def RidgeFit3D(X, Y):
    A = np.zeros((3 * len(X), 4))
    B = np.zeros((3 * len(Y)))
    for i in range((len(X))):
        A[3 * i, :] = np.array([X[i, 0], 1, 0, 0])
        A[3 * i + 1, :] = np.array([X[i, 1], 0, 1, 0])
        A[3 * i + 2, :] = np.array([X[i, 2], 0, 0, 1])
        B[3 * i] = Y[i, 0]
        B[3 * i + 1] = Y[i, 1]
        B[3 * i + 2] = Y[i, 2]
    ATA = np.linalg.inv((np.matmul(A.transpose(), A)))
    ATB = np.matmul(A.transpose(), B)
    ans = np.matmul(ATA, ATB)
    return ans[0],ans[1:]

def CreateObj(points,name):
    path = "data/face.obj"
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    f = open("tmp/"+name+".obj","w")
    index=0
    for i in range(len(content)):
        tmp = content[i].split(" ")
        if (tmp[0] == "v"):
            f.write("v "+ str(points[index,0])+" "+str(points[index,1])+" "+str(points[index,2])+"\n")
            index+=1
        else:
            f.write(content[i]+"\n")
    f.close()

def Landmark(points):
    return points[key_index,:]