#!/usr/bin/env python

"""

"""

import collections
import os
import sys
import tarfile
from urllib import request

from data import DATA_DIR
import numpy as np
import scipy.io
import tensorflow as tf
from tqdm import trange

PROJECT_DIR = '/project/dir' #TODO - Input needed
sys.path.append(PROJECT_DIR)

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
}

def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw

def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        temp_f_name = request.urlretrieve(URLS['svhn'].format(split), None)[0]
        data_dict = scipy.io.loadmat(temp_f_name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits

def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])
    temp_f_name = request.urlretrieve(URLS['cifar10'], None)[0]
    tar = tarfile.open(name=temp_f_name)
    train_data_batches, train_data_labels = [], []
    for batch in range(1, 6):
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
        train_data_batches.append(data_dict['data'])
        train_data_labels.append(data_dict['labels'].flatten())
    train_set = {'images': np.concatenate(train_data_batches, axis=0),
                 'labels': np.concatenate(train_data_labels, axis=0)}
    data_dict = scipy.io.loadmat(tar.extractfile(
        'cifar-10-batches-mat/test_batch.mat'))
    test_set = {'images': data_dict['data'],
                'labels': data_dict['labels'].flatten()}
    
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    print('Saved:', filename)

def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not os.path.exists(filename):
            return False
    return True

def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)
    for filename, contents in files.items():
        with open(os.path.join(DATA_DIR, filename), 'w') as f:
            f.write(contents)

def _is_installed_folder(name, folder):
    return os.path.exists(os.path.join(DATA_DIR, name, folder))

CONFIGS = dict(
    cifar10=dict(loader=_load_cifar10,
                 checksums=dict(train=None, test=None)),
    svhn=dict(loader=_load_svhn,
              checksums=dict(train=None, test=None, extra=None)))

if __name__ == '__main__':
    if len(sys.argv[1:]):
        subset = set(sys.argv[1:])
    else:
        subset = set(CONFIGS.keys())
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with open(filename, 'w') as f:
                    f.write(data)
            elif sub_name == 'files':
                for file_and_data in data:
                    path = os.path.join(DATA_DIR, file_and_data.filename)
                    open(path, "wb").write(file_and_data.data)
            else:
                saver(data, '%s-%s' % (name, sub_name))
