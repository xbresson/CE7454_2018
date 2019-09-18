import tensorflow as tf
import numpy as np
import Network.ults as ults
import Network.resnet
import os
import cv2
import dlib


from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Model():
    def Load(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        X = tf.placeholder(tf.float32, [None, 224, 224, 3], name='Input_Images')
        SHAPE = tf.placeholder(tf.float32, [None, 100], name='SHAPE')
        EXP = tf.placeholder(tf.float32, [None, 79], name='EXP')
        EULAR = tf.placeholder(tf.float32, [None, 3], name='EULAR')
        T = tf.placeholder(tf.float32, [None, 2], name='T')
        S = tf.placeholder(tf.float32, [None], name='S')

        # Build model
        hp = Network.resnet.HParams(batch_size=256,
                            num_gpus=1,
                            num_output=185,
                            weight_decay=0.0001,
                            momentum=0.9,
                            finetune=False)

        network_train = Network.resnet.ResNet(hp, X, SHAPE, EXP, EULAR, T, S, global_step, name="train")
        network_train.build_model()
        network_train.build_train_op()
        train_summary_op = tf.summary.merge_all()  # Summaries(training)
        print('Number of Weights: %d' % network_train._weights)
        print('FLOPs: %d' % network_train._flops)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
        print("sess 0")
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8),
            allow_soft_placement=False,
            # allow_soft_placement=True,
            log_device_placement=False))
        print("sess 1")
        sess.run(init)
        print("sess done")
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

        #Load pre-trained model

        checkpoint_dir = "train"  # os.path.join(export_dir, 'checkpoint')
        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoints and checkpoints.model_checkpoint_path:
            checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, checkpoints_name))
        print('Load checkpoint %s' % checkpoints_name)

        init_step = global_step.eval(session=sess)


        # Face detector
        predictor_path = 'Network/shape_predictor_68_face_landmarks.dat'
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.mean_data = np.array(np.load("Input/mean_data.npy"),dtype = np.float16)
        self.model = network_train
        self.sess = sess
        self.X = X
        self.SHAPE = SHAPE
        self.EXP = EXP
        self.EULAR = EULAR
        self.T = T
        self.S = S
        self.points = ults.CalculateDense(np.zeros([100]), np.zeros([79]))
    def OneFrame(self,resized_img):
        batch_data =np.resize( (resized_img- self.mean_data)/255.0,(1,224,224,3))
        batch_labels = np.zeros((1,185))
        eular_logits, t_logits, s_logits = self.sess.run(
            [self.model.eular_logits, self.model.t_logits,
             self.model.s_logits], \
            feed_dict={self.model.is_train: False, self.X: batch_data,
                       self.SHAPE: batch_labels[:, :100], self.EXP: batch_labels[:, 100:179],
                       self.EULAR: batch_labels[:, 179:182], self.T: batch_labels[:, 182:184],
                       self.S: batch_labels[:, 184]})
        return np.array(eular_logits[0, :]),np.array(t_logits[0, :]),np.array(s_logits[0, :])

    def Landmark(self,shape):
        tmp = np.zeros((68, 2), dtype=np.uint)
        for i in range(68):
            tmp[i, 0] = shape.part(i).x
            tmp[i, 1] = shape.part(i).y
        return tmp

    def Normalize(self,img):
        gray_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray_, 1)

        shape = self.predictor(gray_, dets[0])
        landmark_ = self.Landmark(shape)
        xmin = np.min(landmark_[:, 0])
        xmax = np.max(landmark_[:, 0])
        ymin = np.min(landmark_[:, 1])
        ymax = np.max(landmark_[:, 1])
        old_cx = (xmin + xmax) / 2
        old_cy = (ymin + ymax) / 2;
        cx = (224 - 1) / 2.0
        cy = (224 - 1) * 2.0 / 5.0;
        length = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
        length *= 1.2
        ori_crop_scale = 224 / length

        image = cv2.resize(img, (0, 0), fx=ori_crop_scale, fy=ori_crop_scale)
        old_cx = old_cx * ori_crop_scale
        old_cy = old_cy * ori_crop_scale

        start_x = int(old_cx - cx)
        start_y = int(old_cy - cy)
        crop_image = image[start_y:start_y + 224, start_x:start_x + 224]
        shape_ = np.shape(crop_image)
        resized_img = np.zeros((224, 224, 3), dtype=np.uint8)
        resized_img[:shape_[0], :shape_[1], :] = crop_image
        return resized_img



    def Create3D(self,eular,scale):
        R = ults.eulerAnglesToRotationMatrix(eular)
        points = np.matmul(self.points, R)
        ults.CreateObj(points / 10000, "test" )


# FacePoseNet = Model()
# FacePoseNet.Load()
# type in the input image path

# file_name = ['gx','yj','zj']
# sub_file_name = ['GT']
# for i_file in range(len(file_name)):
#     for i_sub_file in range(len(sub_file_name)):
#         path = './DL_data/%s/%s'%(file_name[i_file],sub_file_name[i_sub_file])
#         f = open("./DL_data/%s/%s.txt" %(file_name[i_file],sub_file_name[i_sub_file]), 'w+')
#         arr = os.listdir(path)
#         for i_frame in range(len(arr)):
#             try:
#                 print(os.path.join(path,'%d.jpg'%(i_frame)))
#                 test_img = cv2.imread(os.path.join(path,'%d.jpg'%(i_frame)))
#                 resized_img = FacePoseNet.Normalize(img=test_img)
#                 eular, t, s = FacePoseNet.OneFrame(resized_img=resized_img)
#                 #FacePoseNet.Create3D(eular=eular, scale=s)
#                 f.write('frame_%d: %f %f %f %f %f %f' %(i_frame,eular[0],eular[1],eular[2],t[0],t[1],s[0]))
#                 f.write('\n')
#             except:
#                 continue
#
#         f.close()



# test_img = cv2.imread("56.jpg")
# resized_img = FacePoseNet.Normalize(img=test_img)
# eular,t,s = FacePoseNet.OneFrame(resized_img=resized_img)
# FacePoseNet.Create3D(eular = eular,scale=s)
#
# a = tf.Variable([123])
#
# init = tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     s=sess.run([a])