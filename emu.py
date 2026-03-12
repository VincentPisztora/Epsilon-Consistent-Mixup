# -*- coding: utf-8 -*-

import sys

import numpy as np
import tensorflow as tf

import time
import functools
from tqdm import trange, tqdm

import os
import json
from collections import defaultdict
from utils import get_config

###############################################################################
###############################################################################
###############################################################################
### Inputs
###############################################################################
###############################################################################
###############################################################################

_GPUS = None

data_dir = 'data/dir' #TODO - Input needed
models_dir = 'model/outputs/dir' #TODO - Input needed

data_name = str(sys.argv[1]) #'cifar10', 'svhn'
seed = int(sys.argv[2]) #0, 1, 2
n_label = int(sys.argv[3]) #40, 250
n_valid = int(sys.argv[4]) #1000
aug = str(sys.argv[5]) #'y'
method = str(sys.argv[6]) #'mu', 'emu'
arch = str(sys.argv[7]) #'CNN13', 'WideResNet28_2'

iters = 400*(2**10) #409600
warmup = 2**14 #16384
report_freq = 2**10 #1024
checkpoint_freq = 2**10 #1024
batch_size = 2**6 #64

lr = float(sys.argv[12]) #0.002
ema = float(sys.argv[13]) #0.999

wd = float(sys.argv[8])*lr #0.06, 0.12, 0.18, 0.20, 0.30, 0.50, 0.60, 0.90
w_u = float(sys.argv[9]) #1, 10, 20, 50, 100
beta = float(sys.argv[10]) #0.1, 0.2, 0.5, 1.0
eps = float(sys.argv[11]) #0.0, 10.0

model_id = str('_method_'+method+'_aug_'+str(aug)+'_arch_'+arch+'_iters_'+str(iters)+'_warmup_'+str(warmup)+
                        '_lr_'+str(lr)+'_batch_size_'+str(batch_size)+'_ema_'+str(ema)+
                        '_wd_'+str(wd)+'_wu_'+str(w_u)+'_beta_'+str(beta)+'_eps_'+str(eps))

###############################################################################
###############################################################################
###############################################################################
### Make semi-supervised data splits
###############################################################################
###############################################################################
###############################################################################

data_prefix = '%s.%d@%d' % (data_name, seed, n_label) 

checkpoint_suffix = str('checkpoints_'+data_prefix+model_id)
checkpoint_dir = os.path.join(models_dir,checkpoint_suffix)
os.makedirs(checkpoint_dir, exist_ok=True)

data_output_path = checkpoint_dir
input_files = [os.path.join(data_dir, data_name+'-train.tfrecord')]

def get_class(serialized_example):
    return tf.parse_single_example(serialized_example, features={'label': tf.FixedLenFeature([], tf.int64)})['label']

count = 0
id_class = []
class_id = defaultdict(list) 
print('Computing class distribution')
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(get_class, 4).batch(1 << 10)
it = dataset.make_one_shot_iterator().get_next()

try:
    with tf.Session(config=get_config()) as session, tqdm(leave=False) as t:
        while 1:
            old_count = count
            for i in session.run(it): 
                id_class.append(i) 
                class_id[i].append(count) 
                count += 1
            t.update(count - old_count) 
except tf.errors.OutOfRangeError:
        pass
print('%d records found' % count)

nclass = len(class_id)
train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64) 
train_stats /= train_stats.max() 
print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))

class_id = [np.array(class_id[i], dtype=np.int64) for i in range(nclass)] 

np.random.seed(seed)
for i in range(nclass):
    np.random.shuffle(class_id[i])

npos = np.zeros(nclass, np.int64)
label = []
for i in range(n_label):
    c = np.argmax(train_stats - npos / max(npos.max(), 1))
    label.append(class_id[c][npos[c]])
    npos[c] += 1

del npos, class_id

label = frozenset([int(x) for x in label])

print('Creating split in %s' % data_prefix)

npos = np.zeros(nclass, np.int64)
class_data = [[] for _ in range(nclass)]
unlabel = []

os.makedirs(data_output_path, exist_ok=True) 
os.chdir(data_output_path) 
with tf.python_io.TFRecordWriter(data_prefix + '-label.tfrecord') as writer_label, tf.python_io.TFRecordWriter(data_prefix + '-unlabel.tfrecord') as writer_unlabel:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files: 
            for record in tf.python_io.tf_record_iterator(input_file): 
                if pos in label:
                    writer_label.write(record) 
                else: 
                    class_data[id_class[pos]].append((pos, record)) 
                    while True:
                        c = np.argmax(train_stats - npos / max(npos.max(), 1))
                        if class_data[c]: 
                            p, v = class_data[c].pop(0) 
                            unlabel.append(p)
                            writer_unlabel.write(v)
                            npos[c] += 1 
                        else:
                            break
                pos += 1
                loop.update()
        for remain in class_data:
            for p, v in remain:
                unlabel.append(p)
                writer_unlabel.write(v)
        loop.close()
with open(data_prefix + '-map.json', 'w') as writer:
    writer.write(json.dumps(
            dict(label=sorted(label), unlabel=unlabel), indent=2, sort_keys=True))

###############################################################################
###############################################################################
###############################################################################
### Data loading functions
###############################################################################
###############################################################################
###############################################################################

def memoize(dataset: tf.data.Dataset, shuffle_buffer) -> tf.data.Dataset:
    data = []
    with tf.Session(config=get_config()) as session:
        dataset = dataset.prefetch(64)
        it = dataset.make_one_shot_iterator().get_next()
        try:
            while 1:
                data.append(session.run(it))
        except tf.errors.OutOfRangeError:
            pass
    observations = np.stack([x['observation'] for x in data])
    labels = np.stack([x['label'] for x in data])
    
    def tf_get(index):
        def get(index):
            return observations[index], labels[index]
        
        observation, label = tf.py_func(get, [index], [tf.float32, tf.int64])
        return dict(observation=observation, label=label)
    
    dataset = tf.data.Dataset.range(len(data)).repeat()
    dataset = dataset.shuffle(len(data) if len(data) < shuffle_buffer else shuffle_buffer) 
    return dataset.map(tf_get)

def record_parse(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'label': tf.FixedLenFeature([], tf.int64)})
    observation = tf.image.decode_image(features['image'])
    observation = tf.cast(observation, tf.float32) * (2.0 / 255) - 1.0
    label = features['label']
    return dict(observation=observation, label=label)

def createDataSplits(prefix,data_dir_train,data_dir_test,record_parse,augment,seed,n_label,n_valid,para=4,do_memoize=True,shuffle_buffer=8192):
    fullname = (prefix+'.%d@%d') % (seed, n_label)
    root = os.path.join(data_dir_train, fullname)
    fn = functools.partial(memoize,shuffle_buffer=shuffle_buffer) if do_memoize else lambda x: x.repeat().shuffle(shuffle_buffer)
    
    train_labeled = tf.data.TFRecordDataset(root + '-label.tfrecord').map(record_parse,num_parallel_calls=para)
    train_unlabeled = tf.data.TFRecordDataset(root + '-unlabel.tfrecord').map(record_parse,num_parallel_calls=para).skip(n_valid)
    
    train_labeled = fn(train_labeled).map(augment[0], para)
    train_unlabeled = fn(train_unlabeled).map(augment[0], para)
    
    return dict(name=fullname + '-' + str(n_valid),
                train_labeled=train_labeled,
                train_unlabeled=train_unlabeled,
                eval_labeled=tf.data.TFRecordDataset(root + '-label.tfrecord').map(record_parse,num_parallel_calls=para),
                eval_unlabeled=tf.data.TFRecordDataset(root + '-unlabel.tfrecord').map(record_parse,num_parallel_calls=para).skip(n_valid),
                valid=tf.data.TFRecordDataset(root + '-unlabel.tfrecord').map(record_parse,num_parallel_calls=para).take(n_valid),
                test=tf.data.TFRecordDataset(os.path.join(data_dir_test, '%s-test.tfrecord' % prefix)).map(record_parse,num_parallel_calls=para))

def augment_mirror(x):
    return tf.image.random_flip_left_right(x)

def augment_shift(x, w):
    y = tf.pad(x, [[w] * 2, [w] * 2, [0] * 2], mode='REFLECT')
    return tf.random_crop(y, tf.shape(x))

augment_cifar10 = lambda x: dict(observation=augment_shift(augment_mirror(x['observation']), 4), label=x['label'])
augment_svhn = lambda x: dict(observation=augment_shift(x['observation'], 4), label=x['label'])

###############################################################################
###############################################################################
###############################################################################
### Model
###############################################################################
###############################################################################
###############################################################################

def model_vars(scope=None):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

class Model():
    def __init__(self,arch,nclass,obs_shape,ema,wd,lr,w_u,warmup,eps,dataset,batch_size,beta,method):
        self.sess = None
        self.step = tf.train.get_or_create_global_step()
        self.classifier = self.def_classifier(arch=arch,nclass=nclass)
        self.ops = dict(update_op=tf.assign_add(self.step,1))
        self.ops.update(self.def_ops(nclass=nclass,obs_shape=obs_shape,
                                     ema=ema,wd=wd,lr=lr,w_u=w_u,
                                     warmup=warmup,eps=eps,
                                     beta=beta,method=method))
        
        self.print_queue = []
        
        print(' Model '.center(80, '-'))
        to_print = [tuple(['%s' % x for x in (v.name, np.prod(v.shape), v.shape)]) for v in model_vars('classify')]
        to_print.append(('Total', str(sum(int(x[1]) for x in to_print)), ''))
        sizes = [max([len(x[i]) for x in to_print]) for i in range(3)]
        fmt = '%%-%ds  %%%ds  %%%ds' % tuple(sizes)
        for x in to_print[:-1]:
            print(fmt % x)
        print()
        print(fmt % to_print[-1])
        print('-' * 80)
        
    def cache_eval_data(self, dataset):
        
        def collect(dataset):
            it = dataset.batch(1).prefetch(16).make_one_shot_iterator().get_next()
            observations, labels = [], []
            while 1:
                try:
                    v = self.sess.run(it)
                except tf.errors.OutOfRangeError:
                    break
                observations.append(v['observation'])
                labels.append(v['label'])
                
            observations = np.concatenate(observations, axis=0)
            labels = np.concatenate(labels, axis=0)
            
            return {'observations':observations, 'labels':labels}
        
        return {'eval_labeled':collect(dataset['eval_labeled']),
                'valid':collect(dataset['valid']),
                'test':collect(dataset['test'])}
        
    def def_classifier(self,arch,nclass):
            
        if(arch=='WideResNet28_2'):
            def classifier(x, training, scales=3, filters=32, repeat=4, getter=None, **kwargs):
                del kwargs
                leaky_relu = functools.partial(tf.nn.leaky_relu, alpha=0.1)
                bn_args = dict(training=training, momentum=0.999)
                
                def conv_args(k, f):
                    return dict(padding='same',
                                kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5 * k * k * f)))
                
                def residual(x0, filters, stride=1, activate_before_residual=False):
                    x = leaky_relu(tf.layers.batch_normalization(x0, **bn_args))
                    if activate_before_residual:
                        x0 = x
                    
                    x = tf.layers.conv2d(x, filters, 3, strides=stride, **conv_args(3, filters))
                    x = leaky_relu(tf.layers.batch_normalization(x, **bn_args))
                    x = tf.layers.conv2d(x, filters, 3, **conv_args(3, filters))
                    
                    if x0.get_shape()[3] != filters:
                        x0 = tf.layers.conv2d(x0, filters, 1, strides=stride, **conv_args(1, filters))
                    
                    return x0 + x
                
                with tf.variable_scope('classify', reuse=tf.AUTO_REUSE, custom_getter=getter):
                    y = tf.layers.conv2d((x - 0.0) / 1.0, 16, 3, **conv_args(3, 16))
                    for scale in range(scales):
                        y = residual(y, filters << scale, stride=2 if scale else 1, activate_before_residual=scale == 0)
                        for i in range(repeat - 1):
                            y = residual(y, filters << scale)
                    
                    y = leaky_relu(tf.layers.batch_normalization(y, **bn_args))
                    y = tf.reduce_mean(y, [1, 2])
                    logits = tf.layers.dense(y, nclass, kernel_initializer=tf.glorot_normal_initializer())
                return logits
        return classifier
    
    def def_ops(self,nclass,obs_shape,ema,wd,lr,w_u,warmup,eps,beta,method):
        x_in = tf.placeholder(tf.float32, [None] + obs_shape, 'x')
        y_in = tf.placeholder(tf.float32, [None] + obs_shape, 'y')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        w_u *= tf.clip_by_value(tf.cast(self.step, tf.float32) / warmup, 0, 1)
        
        
        l = tf.one_hot(l_in, nclass)
        
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        logits_x = self.classifier(x_in, training=True)
        post_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if v not in skip_ops] 
        
        #######################################################################
        def getter_ema(ema, getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            ema_var = ema.average(var)
            return ema_var if ema_var else var
        
        def model_vars(scope=None):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        
        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(model_vars())
        ema_getter = functools.partial(getter_ema, ema)
        #######################################################################
        
        if(method=='emu'):
            with tf.variable_scope('epsilon', reuse=tf.AUTO_REUSE, custom_getter=None):
                eps = tf.math.abs(tf.Variable(initial_value=eps, trainable=True))
        
        
        nu = 1e-10
        
        l_y = tf.stop_gradient(tf.nn.softmax(self.classifier(y_in, training=True, getter=ema_getter)))
        l_xy = tf.concat([l,l_y],0)
        xy = tf.concat([x_in,y_in],0)
        
        n_x = tf.shape(x_in)[0]
        n = tf.shape(xy)[0]
        
        index = tf.random_shuffle(tf.range(n))
        l_xy_s = tf.gather(l_xy, index)
        xy_s = tf.gather(xy, index)
        
        eta = tf.distributions.Beta(beta, beta).sample([n, 1, 1, 1])
        
        eta = tf.minimum(eta, 1 - eta)
        
        xy_mu = xy*eta + xy_s*(1-eta)
        
        def get_lmbda(eta,eps_scaled):
            lmbda = eta[:,:,0,0]
            lmbda = (lmbda-eps_scaled)/(1.-2*eps_scaled)
            lmbda = tf.clip_by_value(lmbda, 0., 1.)
            return lmbda
        
        xy_flat = tf.reshape(xy,[n,-1])
        xy_s_flat = tf.reshape(xy_s,[n,-1])
        d = tf.sqrt(tf.reduce_sum(input_tensor=(xy_flat-xy_s_flat)**2,axis=1,keepdims=True))+nu
        eps_scaled = tf.minimum(eps/d,.5-nu)
        
        lmbda = get_lmbda(eta,eps_scaled)
        l_xy_mu = l_xy*lmbda + l_xy_s*(1-lmbda)
        
        x_mu = xy_mu[0:n_x]
        y_mu = xy_mu[n_x:]
        
        logits_x_mu = self.classifier(x_mu, training=True)
        logits_y_mu = self.classifier(y_mu, training=True)
        logits_mu = tf.concat([logits_x_mu,logits_y_mu],0)
        
        loss_u = tf.reduce_mean(tf.reduce_sum((tf.nn.softmax(logits_mu)-l_xy_mu)**2,1))
        loss_s = tf.losses.softmax_cross_entropy(onehot_labels=l, logits=logits_x)
        loss_s = tf.reduce_mean(loss_s)
        
        #######################################################################
        #######################################################################
        
        loss = loss_s + w_u*loss_u
        
        post_ops.append(ema_op)
        post_ops.extend([tf.assign(v, v * (1 - wd)) for v in model_vars('classify') if 'kernel' in v.name])
        
        opt = tf.train.AdamOptimizer(lr)
        train_op = opt.minimize(loss, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)
        
        classify_op = tf.nn.softmax(self.classifier(x_in, getter=ema_getter, training=False))
        
        return  {'x_in':x_in,'y_in':y_in,'l_in':l_in,'loss':loss,'loss_s':loss_s,'loss_u':loss_u,
                'train_op':train_op,'classify_op':classify_op}
        
    def train(self,dataset,batches,batch_size,report_freq,checkpoint_freq,checkpoint_dir):  
        
        with tf.Session(config=get_config()) as sess:
            self.sess = sess
            self.cache = self.cache_eval_data(dataset=dataset)
        
        train_labeled = dataset['train_labeled'].batch(batch_size).prefetch(128)
        train_labeled = train_labeled.make_one_shot_iterator().get_next()
        train_unlabeled = dataset['train_unlabeled'].batch(batch_size).prefetch(128)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        
        loss_list = []
        loss_s_list = []
        loss_u_list = []
        acc_list = []
        eps_list = []
        
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=1))
        
        with tf.train.MonitoredTrainingSession(scaffold=scaffold,
                                               save_checkpoint_steps=checkpoint_freq,
                                               checkpoint_dir=checkpoint_dir,
                                               save_summaries_steps=checkpoint_freq,
                                               config=get_config()) as train_session:
            
            self.sess = train_session._tf_sess()
            self.step = self.sess.run(self.step)
            
            
            while self.step < batches:
                eff_report_freq = min(report_freq,batches-self.step)
                loop = range(0,eff_report_freq,1)
                t0 = time.time()
                for i in loop:
                    
                    x, y = self.sess.run([train_labeled, train_unlabeled])
                    _, loss, loss_s, loss_u, self.step = train_session.run([self.ops['train_op'], self.ops['loss'], self.ops['loss_s'], self.ops['loss_u'], self.ops['update_op']],
                                             feed_dict={self.ops['x_in']: x['observation'],
                                                        self.ops['y_in']: y['observation'],
                                                        self.ops['l_in']: x['label']})
                    current_eps = self.sess.run(model_vars('epsilon'))
                    
                    if(i==int(eff_report_freq-1)):
                        loss_list.append(loss)
                        loss_s_list.append(loss_s)
                        loss_u_list.append(loss_u)
                        eps_list.append(current_eps)
                        
                ################################################################################ 
                
                accuracies = []
                for subset in ('eval_labeled', 'valid', 'test'):
                    observations = self.cache[subset]['observations']
                    labels = self.cache[subset]['labels']
                    predicted = []
                    
                    for i in range(0, observations.shape[0], batch_size):
                        p = self.sess.run(self.ops['classify_op'], feed_dict={self.ops['x_in']: observations[i:i + batch_size]})
                        predicted.append(p)
                    predicted = np.concatenate(predicted, axis=0)
                    accuracies.append((predicted.argmax(1) == labels).mean() * 100)
                
                acc_list.append(np.array(accuracies,'f'))
                print('eps:', self.sess.run(model_vars('epsilon')), 'lab-val-test:', accuracies)
                
                ################################################################################
                
                acc_file_path = os.path.join(checkpoint_dir,'accuracies_eval_labeled_valid_test.csv')
                eps_file_path = os.path.join(checkpoint_dir,'epsilon.csv')
                loss_file_path = os.path.join(checkpoint_dir,'losses_tsu.csv')
                if (not os.path.exists(acc_file_path)):
                    np.savetxt(acc_file_path, acc_list, delimiter=",")
                else:
                    acc_temp = np.loadtxt(acc_file_path, delimiter=',')
                    acc_temp = np.vstack((acc_temp,acc_list[-1]))
                    np.savetxt(acc_file_path, acc_temp, delimiter=",")
                
                if (not os.path.exists(eps_file_path)):
                    np.savetxt(eps_file_path, eps_list, delimiter=",")
                else:
                    eps_temp = np.loadtxt(eps_file_path, delimiter=',')
                    eps_temp = np.hstack((eps_temp,eps_list[-1]))
                    np.savetxt(eps_file_path, eps_temp, delimiter=",")
                
                if (not os.path.exists(loss_file_path)):
                    np.savetxt(loss_file_path, np.vstack((loss_list,loss_s_list,loss_u_list)).T, delimiter=",")
                else:
                    loss_temp = np.loadtxt(loss_file_path, delimiter=',')
                    current_losses = np.array([loss_list[-1],loss_s_list[-1],loss_u_list[-1]])
                    loss_temp = np.vstack((loss_temp, current_losses))
                    np.savetxt(loss_file_path, loss_temp, delimiter=",")
                
                ################################################################################ 
                
                t1 = time.time()
                print('1 epoch time', np.round(t1-t0,2), 'sec | proj total runtime:',np.round((t1-t0)*400/(60**2),2),
                      'hrs | proj remaining runtime:', np.round((t1-t0)*(400-(self.step/report_freq))/(60**2),2), 'hrs')
                
        return(loss_list,loss_s_list,loss_u_list)
        
    def eval_latest_checkpoint(self, batch_size, checkpoint_dir, sess):
        ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        
        accuracies = []
        for subset in ('eval_labeled', 'valid', 'test'):
            observations = self.cache[subset]['observations']
            labels = self.cache[subset]['labels']
            predicted = []
            
            for i in range(0, observations.shape[0], batch_size):
                p = sess.run(self.ops['classify_op'], feed_dict={self.ops['x_in']: observations[i:i + batch_size]})
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)
        print(np.array(accuracies))
        return np.array(accuracies)
    
    def eval_stats(self, batch_size):        
        accuracies = []
        for subset in ('eval_labeled', 'valid', 'test'):
            observations = self.cache[subset]['observations']
            labels = self.cache[subset]['labels']
            predicted = []
            
            for i in range(0, observations.shape[0], batch_size):
                p = self.sess.run(self.ops['classify_op'], feed_dict={self.ops['x_in']: observations[i:i + batch_size]})
                predicted.append(p)
            predicted = np.concatenate(predicted, axis=0)
            accuracies.append((predicted.argmax(1) == labels).mean() * 100)
        
        self.print_queue.append(np.array(accuracies,'f'))
        
        return np.array(accuracies,'f')
    
    def add_summaries(self, batch_size):
        
        def gen_stats():
            return self.eval_stats(batch_size=batch_size)
        
        accuracies = tf.py_func(gen_stats, [], tf.float32)
        
        tf.summary.scalar('accuracy/eval_labeled', accuracies[0])
        tf.summary.scalar('accuracy/valid', accuracies[1])
        tf.summary.scalar('accuracy/test', accuracies[2])
    
###############################################################################
###############################################################################
###############################################################################
### Semi-supervised data loading and model training
###############################################################################
###############################################################################
###############################################################################

tf.reset_default_graph()

if(aug=='y' and data_name=='cifar10'):
    augment = [augment_cifar10]
elif(aug=='y' and data_name=='svhn'):
    augment = [augment_svhn]
else:
    augment = []

data = createDataSplits(prefix=data_name,data_dir_train=checkpoint_dir,data_dir_test=data_dir,record_parse=record_parse,
                        augment=augment,seed=seed,n_label=n_label,n_valid=n_valid,
                        para=4,do_memoize=True,shuffle_buffer=8192)
d_name = data['name']

model = Model(arch=arch,nclass=10,obs_shape=[32,32,3],ema=ema,wd=wd,lr=lr,
              w_u=w_u,warmup=warmup,eps=eps,dataset=data,batch_size=batch_size,
              beta=beta,method=method)

loss, loss_s, loss_u, = model.train(dataset=data, batches=iters, batch_size=batch_size, 
                                   report_freq=report_freq, checkpoint_freq=checkpoint_freq, checkpoint_dir=checkpoint_dir)

accs = model.eval_latest_checkpoint(batch_size=128, checkpoint_dir=checkpoint_dir, sess=tf.Session(config=get_config()))

np.savetxt('accs_'+d_name+model_id+'.csv', accs, delimiter=",")

print(model_id)

