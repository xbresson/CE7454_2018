import numpy as np
import dlib
import glob
import cv2
import os

os.chdir('/media/imi-yujun/579e63e9-5852-43a7-9323-8e51241f5d3a/yujun/Course_porject_fac e')

def LoadBase():
    predictor_path = 'Network/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return predictor,detector

def SaveLandmark(shape):
    tmp = np.zeros((68,2),dtype=np.uint)
    for i in range(68):
        tmp[i,0] = shape.part(i).x
        tmp[i, 1] = shape.part(i).y
    return tmp

def Run():
    count =0
    predictor, detector  = LoadBase()
    image_list = glob.glob('CoarseData/CoarseData/*/*.jpg')
    for path_ in image_list:

        image_= cv2.imread(path_)
        #gray_= cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
        try:
            print(count)
            count += 1
            # dets = detector(gray_, 1)
            # shape = predictor(gray_, dets[0])
            # tmp = SaveLandmark(shape)
            # np.savetxt(path_[:len(path_)-4]+'_landmark.txt',tmp, fmt='%d')
            # res = Normalize(image_,tmp)
            # cv2.imwrite(path_[:len(path_)-4]+'_224.png',res)
            #
            landmark = np.loadtxt(path_[:len(path_)-4]+'_landmark.txt')
            shape_, exp_, eular_, translate_, scale_ = Get3Dmm(path_[:len(path_)-4]+'.txt')
            crop_image, translation, new_scale = Normalize2(image_,landmark,translate_,scale_)
            cv2.imwrite(path_[:len(path_) - 4] + '_224.png', crop_image)
            label = np.zeros([185])
            label[:100] = shape_
            label[100:179] = exp_
            label[179:182] =eular_
            label[182:184] = translate_
            label[184] = scale_
            np.savetxt(path_[:len(path_) - 4] + '_224.txt',label)

        except:
            continue
            print("error")

def Normalize(image,landmark_):
    xmin = np.min(landmark_[:,0])
    xmax = np.max(landmark_[:,0])
    ymin = np.min(landmark_[:,1])
    ymax = np.max(landmark_[:,1])
    sub_image = image[ymin:ymax,xmin:xmax]
    res = cv2.resize(sub_image, (224,224), interpolation=cv2.INTER_LINEAR)
    return res


def Package():
    Save_path = 'Input/'
    image_list = glob.glob('CoarseData/CoarseData/*/*_224.png')
    data = np.zeros((len(image_list),224,224,3),dtype=np.uint8)
    label = np.zeros((len(image_list),185))
    for i in range(len(image_list)):
        print(i)
        path_ = image_list[i]
        img = cv2.imread(path_)
        # f = open(path_[:len(path_)-8]+'.txt')
        # content = f.readlines()
        # content= [x.strip() for x in content]
        # label[i,:100] = np.array(content[0].split(" "),dtype = np.float)
        # label[i,100:179] = np.array(content[1].split(" "),dtype = np.float)
        # label[i,179:] = np.array(content[2].split(" "),dtype = np.float)
        # f.close()
        label[i,:] = np.loadtxt(path_[:len(path_)-8]+'_224.txt')
        data[i,:,:] = np.array(img,dtype=np.uint8)

    np.save(Save_path+'data.npy',data)
    np.save(Save_path+'label.npy',label)

def Split():
    test_num = 5000
    data = np.load('Input/data.npy')
    label= np.load('Input/label.npy')
    train_data= data[test_num:,:]
    test_data = data[:test_num,:]
    test_label = label[:test_num,:]
    train_label = label[test_num:,:]
    np.save('Input/train_data.npy',train_data)
    np.save('Input/mean_data.npy',np.mean(data,axis=0))
    np.save('Input/mean_label.npy', np.mean(label, axis=0))
    np.save('Input/test_data.npy',test_data)
    np.save('Input/train_label.npy',train_label)
    np.save('Input/test_label.npy',test_label)
    np.save('Input/std_label.npy', np.std(label, axis=0))



net_img_size = 224
def Normalize2(image, landmark_, translation=np.array([0,0]), scale=0):
    xmin = np.min(landmark_[:, 0])
    xmax = np.max(landmark_[:, 0])
    ymin = np.min(landmark_[:, 1])
    ymax = np.max(landmark_[:, 1])
    old_cx = (xmin + xmax) / 2
    old_cy = (ymin + ymax) / 2;
    cx = (net_img_size - 1) / 2.0
    cy = (net_img_size - 1) * 2.0 / 5.0;
    length = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
    length *= 1.2
    ori_crop_scale = net_img_size / length
    new_scale = scale * ori_crop_scale
    image = cv2.resize(image, (0, 0), fx=ori_crop_scale, fy=ori_crop_scale)
    old_cx = old_cx * ori_crop_scale
    old_cy = old_cy * ori_crop_scale

    start_x = int(old_cx - cx)
    start_y = int(old_cy - cy)
    crop_image = image[start_y:start_y + 224, start_x:start_x + 224]
    shape_ = np.shape(crop_image)
    tmp = np.zeros((224,224,3),dtype=np.uint8)
    tmp[:shape_[0],:shape_[1],:] = crop_image
    translation = translation * ori_crop_scale
    translation[0] = translation[0] - start_x
    translation[1] = translation[1] - (len(image) - 224-start_y)

    # landmark_=landmark_*ori_crop_scale
    # tmp = np.zeros((224,224),dtype=np.uint8)
    # for i in range(68):
    #     tmp[ int(landmark_[i,1] - start_y),int(landmark_[i,0] - start_x)  ] = 255;
    # cv2.imwrite("landmarl.jpg",tmp)

    return tmp, translation, new_scale



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

def Custom():
    cap = cv2.VideoCapture('Input/gx.MOV')
    sample_num = 150
    M = cv2.getRotationMatrix2D((1920 / 2, 1080 / 2), 270, 1)
    predictor, detector = LoadBase()
    index_=0
    data = np.zeros((sample_num,224,224,3))
    for i in range(sample_num):
        ret,frame = cap.read()
        frame = cv2.warpAffine(frame, M, (1920, 1080))
        gray_ = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(gray_, 1)
        shape = predictor(gray_, dets[0])
        tmp = SaveLandmark(shape)
        res,_,_ = Normalize2(frame,tmp)
        data[i,:,:,:] = res
        cv2.imwrite('tmp/gx/'+str(index_)+'.jpg', frame)
        print(index_)
        index_+=1
    cap.release()
    np.save("Input/gx.npy",data)

Custom()
# Run()
# Package()
# Split()