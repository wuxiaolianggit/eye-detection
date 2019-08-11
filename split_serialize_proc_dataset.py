import tensorflow as tf
import numpy as np
import glob
import os
import argparse
from time import time
import json
from utils.misc import save_json

def split_folders_with_variable_example_number(dataDir,trainSplit,valSplit,numTries = 50):
    '''splits the subject folders present in dataDir into train, validation and test subsets
    The objective of this function is to split the dataset ensuring that all the images of a subject fall into a single subset (i.e the images of a subject are not mixed between subsets)
    The train, val, test proportion refers to the number of examples (images). As some subjects may have more images than others, if we simply make a random split of the subject folders, the number of examples in each subset may not match the input split proportions
    The problem is solved by making this random subject split a number of times (numTries) and keeping the best split. This best split will be the one closest to the input split proportions in terms of number of examples in each subset.
    This lazy solution is not perfect, but improves the splitting ensuring that all subject images fall in a single subset'''

    folders = glob.glob(os.path.join(dataDir,'*'))
    lFolders= len(folders)

    testSplit = 1 - trainSplit - valSplit
    lTrain = int(trainSplit * lFolders)
    lVal = int(valSplit * lFolders)
    lTest = lFolders - lTrain - lVal
    subsets =  ['train'] * lTrain + ['val'] * lVal + ['test'] * lTest

    targetSplit = np.array([trainSplit,valSplit,testSplit])
    bestSplitError = np.inf
    for i in range(numTries):
        subsets = np.random.permutation(subsets)
        dataSplitInfo = {'num_examples_train' : 0,
                         'train_examples' : [],
                         'train_folders' : [],
                         'num_examples_val' : 0, 
                         'val_examples' : [],
                         'val_folders' : [],
                         'num_examples_test' : 0, 
                         'test_examples' : [],
                         'test_folders' : []}

        for folder,subset in zip(folders,subsets):
            folderFiles = glob.glob(os.path.join(folder,'*'))
            dataSplitInfo['num_examples_'+subset] += len(folderFiles)
            dataSplitInfo[subset+'_examples'] += folderFiles
            dataSplitInfo[subset+'_folders'] += [folder]
        
        dataSplitInfo['num_examples_total'] = dataSplitInfo['num_examples_train'] + dataSplitInfo['num_examples_val'] + dataSplitInfo['num_examples_test']
        dataSplitInfo['frac_train'] = dataSplitInfo['num_examples_train'] / dataSplitInfo['num_examples_total']
        dataSplitInfo['frac_val'] = dataSplitInfo['num_examples_val'] / dataSplitInfo['num_examples_total']
        dataSplitInfo['frac_test'] = dataSplitInfo['num_examples_test'] / dataSplitInfo['num_examples_total']

        currentSplit = np.array([dataSplitInfo['frac_train'], dataSplitInfo['frac_val'], dataSplitInfo['frac_test']])
        splitError = np.sum(np.abs(targetSplit - currentSplit))
        if splitError < bestSplitError:
            bestSplitError = splitError
            bestSplit = dataSplitInfo

    return bestSplit

def printInPlace(text):
    print('\r'+text+'\t'*15,end='',sep = '')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_info_from_filepath(img_path):
    lx,ly,rx,ry = os.path.basename(img_path).split('.')[0].split('_')[1::]
    return int(lx),int(ly),int(rx),int(ry)

def serialize_example(writer,img_path):
    img_path = str(img_path)
    lx,ly,rx,ry = get_info_from_filepath(img_path)
    img_bytes = tf.io.read_file(img_path)
    img = tf.io.decode_image(img_bytes)
    r,c = img.shape.as_list()[0:2]

    lx = lx / c
    rx = rx / c
    ly = ly / r
    ry = ry / r

    eye_data = np.array([lx,ly,rx,ry],dtype = np.float32)

    data = {'img' : _bytes_feature(img_bytes.numpy()),
            'eye' : _bytes_feature(eye_data.tostring())}

    example = tf.train.Example(features = tf.train.Features(feature=data))
    serial = example.SerializeToString()
    writer.write(serial)

def serialize_dataset(shards,out_dir,subsetName):

    for s,shard in enumerate(shards):
        outTfrPath = os.path.join(out_dir,subsetName+'Data-%d.tfr' % s)
        outInfoPath = os.path.join(out_dir,subsetName+'Data-%d.info' % s)
        infoFile = open(outInfoPath,'w')

        print('Writing tfrecord %d of %d (%s)...\n' % (s,len(shards),subsetName))
        print('Output directory: %s\n' % out_dir)
        with tf.io.TFRecordWriter(outTfrPath) as writer:
            t_init = time()
            t_remain = 0
            l = len(shard)
            
            for i,path in enumerate(shard):
                infoFile.write(path+'\n')
                serialize_example(writer,img_path = path)
                printInPlace('Progress: %d/%d ---- Remaining time: %.4f min' % (i,l,t_remain))
                t_elapsed = time() - t_init
                t_remain = (t_elapsed / (i+1)) * (l-i) / 60

            print('\nSaving tfrecord...')
        print('Done')
        infoFile.close()

if __name__ == '__main__':

    cwd = os.path.dirname(os.path.realpath(__file__))
    procDataDir = cwd + os.sep + 'procData'
    outDir = cwd + os.sep + 'tfrecords'

    trainDir = os.path.join(cwd,'tfrecords','train')
    valDir = os.path.join(cwd,'tfrecords','val')
    testDir = os.path.join(cwd,'tfrecords','test')
    os.makedirs(trainDir)
    os.makedirs(valDir)
    os.makedirs(testDir)

    ##### split data
    dataSplitDict = split_folders_with_variable_example_number(procDataDir,trainSplit = 0.7,valSplit = 0.2,numTries=100)
    save_json(dataSplitDict,os.path.join(outDir,'data_info.json'))

    ##### shuffle data
    np.random.shuffle(dataSplitDict['train_examples'])
    np.random.shuffle(dataSplitDict['val_examples'])
    np.random.shuffle(dataSplitDict['test_examples'])

    ##### create shards. As the dataset is small enough, its okay to create one shard for subset
    trainShards = [ dataSplitDict['train_examples'] ]
    valShards = [ dataSplitDict['val_examples'] ]
    testShards = [ dataSplitDict['test_examples'] ]

    ###### train/val/test data serialization
    serialize_dataset(trainShards,trainDir,'train')
    serialize_dataset(valShards,valDir,'val')
    serialize_dataset(testShards,testDir,'test')
    
