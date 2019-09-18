from collections import namedtuple

import tensorflow as tf
import numpy as np

import Network.utils as utils


HParams = namedtuple('HParams',
                    'batch_size, num_gpus, num_output, weight_decay, '
                     'momentum, finetune')

class ResNet(object):
    def __init__(self, hp, images, SHAPE,EXP,EULAR,T,R, global_step, name=None, reuse_weights=False):
        self._hp = hp # Hyperparameters
        self._images = images # Input images
        self._exps = EXP
        self._shapes = SHAPE
        self._eulars = EULAR
        self._ts = T
        self._rs = R

        self._global_step = global_step
        self._name = name
        self._reuse_weights = reuse_weights
        self.lr = tf.placeholder(tf.float32, name="lr")
        self.is_train = tf.placeholder(tf.bool, name="is_train")
        self._counted_scope = []
        self._flops = 0
        self._weights = 0

        mu_shape = np.load("data/mu_shape_.npy")
        mu_exp = np.load("data/mu_exp_.npy")
        b_exp = np.load("data/b_exp_.npy")
        b_shape = np.load("data/b_shape_.npy")

        label_mean = np.load("Input/mean_label.npy")
        label_std   = np.load("Input/std_label.npy")

        # b_tex = np.load("data/b_tex_.npy")
        # mu_tex = np.load("data/mu_tex_.npy")
        key_index = np.loadtxt('data/keyindex.txt', dtype=np.int).tolist()

        self.mu_exp = tf.constant(mu_exp,shape=[1, 3 * 53215],dtype = tf.float32)
        self.b_exp = tf.constant(b_exp, shape=[79, 3 * 53215],dtype = tf.float32)
        self.b_shape = tf.constant(b_shape, shape=[100, 3 * 53215],dtype = tf.float32)
        self.mu_shape = tf.constant(mu_shape,shape=[1, 3 * 53215],dtype = tf.float32)
        self.label_mean =tf.constant(label_mean,shape=[1,185],dtype = tf.float32)
        self.label_std = tf.constant(label_std, shape=[1,185],  dtype=tf.float32)
    def build_tower(self, images, shapes,exps,eulars,ts,ss):
        print('Building model')

        def Points(exp, shape, eular, t, s):
            tmp = tf.reshape(self.mu_exp + self.mu_shape + tf.matmul(tf.reshape(exp, (1, 79)),self.b_exp) + tf.matmul(
                tf.reshape(shape, (1, 100)), self.b_shape), (53215, 3)) * s

            R_x = tf.reshape(tf.stack([[1, 0, 0],
                                       [0, tf.cos(eular[0]), -tf.sin(eular[0])],
                                       [0, tf.sin(eular[0]), tf.cos(eular[0])]
                                       ]), (3, 3))
            R_y = tf.reshape(tf.stack([[tf.cos(eular[1]), 0, tf.sin(eular[1])],
                                       [0, 1, 0],
                                       [-tf.sin(eular[1]), 0, tf.cos(eular[1])]
                                       ]), (3, 3))
            R_z = tf.reshape(tf.stack([[tf.cos(eular[2]), -tf.sin(eular[2]), 0],
                                       [tf.sin(eular[2]), tf.cos(eular[2]), 0],
                                       [0, 0, 1]
                                       ]), (3, 3))
            R = tf.matmul(R_z, tf.matmul(R_y, R_x))
            tmp = tf.slice(tf.matmul(tmp, R), [0, 0], [-1, 2]) + t
            return tmp

        # filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(images, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        # Logit
        with tf.variable_scope('logits') as scope:
            print('\tBuilding unit: %s' % scope.name)
            x = tf.reduce_mean(x, [1, 2])
            x =self._fc(x, 185)
            logits_ = x*self.label_std+self.label_mean
        logits_shape =   tf.slice(logits_,[0,0],[-1,100]) #@self._fc(x, 100)
        logits_exp= tf.slice(logits_,[0,100],[-1,79])#self._fc(x, 79)
        logits_eular = tf.slice(logits_,[0,179],[-1,3])#self._fc(x, 3)
        logits_t = tf.slice(logits_,[0,182],[-1,2])#self._fc(x, 2)
        logits_s =tf.slice(logits_,[0,184],[-1,1]) #self._fc(x, 1)

        shape_loss = tf.reduce_mean(tf.square((logits_shape - shapes)*0.0001))/10.0
        exp_loss = tf.reduce_mean(tf.square((logits_exp - exps)*0.0001))/10.0
        eular_loss =tf.reduce_mean(tf.square(logits_eular - eulars))*100
        t_loss = tf.reduce_mean(tf.square(logits_t - ts))/10
        s_loss = tf.reduce_mean(tf.abs(logits_s - ss))*1000
        parameter_loss =eular_loss+s_loss+t_loss #+exp_loss+shape_loss

        #proj_pose_points =tf.map_fn( lambda x: Points(x[0],x[1],x[2],x[3],x[4] ),
        #                             [exps,shapes,logits_eular ,logits_t,logits_s],dtype=tf.float32)
        proj_geo_points = tf.map_fn(lambda x: Points(x[0], x[1], x[2], x[3], x[4]),
                                     [logits_exp, logits_shape, eulars ,ts,ss], dtype=tf.float32)

        proj_tru_points = tf.map_fn( lambda x: Points(x[0],x[1],x[2],x[3],x[4] ),[exps,shapes,eulars ,ts,ss],dtype=tf.float32)

        geo_loss = tf.reduce_mean(tf.square((proj_geo_points - proj_tru_points)))
        pose_loss = 0#tf.reduce_mean(tf.square((proj_pose_points - proj_tru_points)))
        lambda_ =0.5# geo_loss/(geo_loss+pose_loss)
        points_loss =(lambda_)*pose_loss+(1-lambda_)*geo_loss
        loss = points_loss+parameter_loss



        # tf.map_fn( lambda x: self.Points(x[0],x[1],x[2],x[3],x[4] ),(logits_exp,logits_shape,logits_eular ,logits_t,logits_s),dtype=tf.float32)
        return [logits_exp,logits_shape,logits_eular ,logits_t,logits_s], loss,  shape_loss, exp_loss,eular_loss,s_loss,t_loss,points_loss,geo_loss,pose_loss





    def build_model(self):
        # Split images and labels into (num_gpus) groups
        images = tf.split(self._images, num_or_size_splits=self._hp.num_gpus, axis=0)
        exps = tf.split(self._exps, num_or_size_splits=self._hp.num_gpus, axis=0)

        eulars = tf.split(self._eulars, num_or_size_splits=self._hp.num_gpus, axis=0)
        shapes = tf.split(self._shapes, num_or_size_splits=self._hp.num_gpus, axis=0)
        ts = tf.split(self._ts, num_or_size_splits=self._hp.num_gpus, axis=0)
        rs = tf.split(self._rs, num_or_size_splits=self._hp.num_gpus, axis=0)


        # Build towers for each GPU
        self._logits_list = []
        self._loss_list = []

        self.shape_logits_list= []
        self.exp_logits_list= []
        self.eular_logits_list= []
        self.t_logits_list= []
        self.s_logits_list= []


        self._shape_loss_list = []
        self._exp_loss_list = []
        self._eular_loss_list = []
        self._s_loss_list = []
        self._t_loss_list = []
        self._points_loss_list = []
        self._geo_loss_list = []
        self._pose_loss_list = []

        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Build a tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()
                    tf_mu_exp = tf.Variable(self.mu_exp,trainable=False)
                    tf_mu_shape = tf.Variable(self.mu_shape, trainable=False)
                    tf_b_exp = tf.Variable(self.b_exp, trainable=False)
                    tf_b_shape = tf.Variable(self.b_shape, trainable=False)

                    logits, loss, shape_loss, exp_loss,eular_loss,s_loss,t_loss,points_loss,geo_loss,pose_loss = self.build_tower(images[i],shapes[i], exps[i],eulars[i],ts[i],rs[i])#,tf_mu_exp,tf_mu_shape,tf_b_exp,tf_b_shape)
                    self.shape_logits_list.append(logits[0])
                    self.exp_logits_list.append(logits[1])
                    self.eular_logits_list.append(logits[2])
                    self.t_logits_list.append(logits[3])
                    self.s_logits_list.append(logits[4])



                    self._loss_list.append(loss)

                    self._shape_loss_list.append(shape_loss)
                    self._exp_loss_list.append(exp_loss)
                    self._eular_loss_list.append(eular_loss)
                    self._s_loss_list.append(s_loss)
                    self._t_loss_list.append(t_loss)
                    self._points_loss_list.append(points_loss)
                    self._geo_loss_list.append(geo_loss)
                    self._pose_loss_list.append(pose_loss)


        # Merge losses, accuracies of all GPUs
        with tf.device('/CPU:0'):
            self.shape_logits = tf.concat(self.shape_logits_list, axis=0)
            self.exp_logits = tf.concat(self.exp_logits_list, axis=0)
            self.eular_logits = tf.concat(self.eular_logits_list, axis=0)
            self.t_logits = tf.concat(self.t_logits_list, axis=0)
            self.s_logits = tf.concat(self.s_logits_list, axis=0)

            self.loss = tf.reduce_mean(self._loss_list, name="mse")
            self.shape_loss = tf.reduce_mean(self._shape_loss_list)
            self.exp_loss= tf.reduce_mean(self._exp_loss_list)
            self.eular_loss = tf.reduce_mean(self._eular_loss_list)
            self.s_loss= tf.reduce_mean(self._s_loss_list)
            self.t_loss=tf.reduce_mean(self._t_loss_list)
            self.points_loss=tf.reduce_mean(self._points_loss_list)

            self.geo_loss=tf.reduce_mean(self._geo_loss_list)
            self.pose_loss=tf.reduce_mean(self._pose_loss_list)

            tf.summary.scalar((self._name+"/" if self._name else "") + "mse", self.loss)




    def build_train_op(self):
        # Learning rate
        tf.summary.scalar((self._name+"/" if self._name else "") + 'learing_rate', self.lr)

        opt = tf.train.MomentumOptimizer(self.lr, self._hp.momentum)
        self._grads_and_vars_list = []

        # Computer gradients for each GPU
        for i in range(self._hp.num_gpus):
            with tf.device('/GPU:%d' % i), tf.variable_scope(tf.get_variable_scope()):
                with tf.name_scope('tower_%d' % i) as scope:
                    print('Compute gradients of tower: %s' % scope)
                    if self._reuse_weights or i > 0:
                        tf.get_variable_scope().reuse_variables()

                    # Add l2 loss
                    costs = [tf.nn.l2_loss(var) for var in tf.get_collection(utils.WEIGHT_DECAY_KEY)]
                    l2_loss = tf.multiply(self._hp.weight_decay, tf.add_n(costs))
                    total_loss = self._loss_list[i] #+ 0.0001*l2_loss

                    # Compute gradients of total loss
                    grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())

                    # Append gradients and vars
                    self._grads_and_vars_list.append(grads_and_vars)

        # Merge gradients
        print('Average gradients')
        with tf.device('/CPU:0'):
            grads_and_vars = self._average_gradients(self._grads_and_vars_list)

            # Finetuning
            if self._hp.finetune:
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if "unit3" in var.op.name or \
                        "unit_last" in var.op.name or \
                        "/q" in var.op.name or \
                        "logits" in var.op.name:
                        print('\tScale up learning rate of % s by 10.0' % var.op.name)
                        grad = 10.0 * grad
                        grads_and_vars[idx] = (grad,var)

            # Apply gradient
            apply_grad_op = opt.apply_gradients(grads_and_vars, global_step=self._global_step)

            # Batch normalization moving average update
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.train_op = tf.group(*(update_ops+[apply_grad_op]))


    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x


    def _average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # If no gradient for a variable, exclude it from output
            if grad_and_vars[0][0] is None:
                continue

            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads


    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)
