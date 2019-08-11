import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  ###### disable tensorflow warnings logging
import tensorflow as tf
import numpy as np
from network import UNet,scriptPath
import glob
from multiprocessing import cpu_count
from time import time
import cv2
import sys
from utils.misc import load_json,save_json,printInPlace,get_time_left,get_current_time
from utils.training import write_summary,lr_decay_schedule
from utils.imgproc import normalization
from shutil import copy
from scipy.ndimage import rotate

def rotate_image(image,rotAngle,interpolationOrder):
    rot_img = rotate(image,angle = rotAngle,reshape=False,order = interpolationOrder)
    return rot_img

def eye_to_heatmap(out_shape,eye):

    lx = int(eye[0] * out_shape[1])
    ly = int(eye[1] * out_shape[0])
    rx = int(eye[2] * out_shape[1])
    ry = int(eye[3] * out_shape[0])
    h1 = np.zeros(out_shape, dtype = np.int8)
    h1 = cv2.circle(h1,center = (lx,ly),radius = 5,color = 1,thickness = -1)
    h2 = np.zeros(out_shape, dtype = np.int8)
    h2 = cv2.circle(h2,center = (rx,ry),radius = 5,color = 1,thickness = -1)
    h3 = np.abs(h1+h2 -1)
    heatmap = np.stack((h1,h2,h3),axis=2).astype(np.uint8)
    return heatmap

def random_crop_and_resize(image,heatmap,min_crop_relative_size,crop_probability):
    if tf.random.uniform(shape = [1],minval = 0,maxval = 1,dtype = tf.float32)[0] < crop_probability:
        orig_shape = image.shape

        crop_rel_size = tf.random.uniform(shape = [1],minval = min_crop_relative_size,maxval = 1,dtype = tf.float32)[0]
        crop_rel_init_pos_limit = 1 - crop_rel_size
        crop_init_pos_rel_r = tf.random.uniform(shape = [1],minval = 0,maxval = crop_rel_init_pos_limit,dtype = tf.float32)[0]
        crop_init_pos_rel_c = tf.random.uniform(shape = [1],minval = 0,maxval = crop_rel_init_pos_limit,dtype = tf.float32)[0]
        crop_end_pos_rel_r = crop_init_pos_rel_r + crop_rel_size
        crop_end_pos_rel_c = crop_init_pos_rel_c + crop_rel_size

        crop_r1 = tf.cast(crop_init_pos_rel_r * orig_shape[0], dtype = tf.int32)
        crop_r2 = tf.cast(crop_end_pos_rel_r * orig_shape[0],dtype = tf.int32)

        crop_c1 = tf.cast(crop_init_pos_rel_c * orig_shape[1],dtype = tf.int32)
        crop_c2 = tf.cast(crop_end_pos_rel_c * orig_shape[1],dtype = tf.int32)

        image = image[crop_r1:crop_r2,crop_c1:crop_c2,:]
        heatmap = heatmap[crop_r1:crop_r2,crop_c1:crop_c2,:]
        image = tf.image.resize(image, orig_shape[0:2])
        heatmap = tf.image.resize(heatmap, orig_shape[0:2],method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    return image,heatmap

def load_dataset(tfrecordDir,input_shape,batch_size,shuffle_buffer_size):

    def _parse_function(proto):
        keys_to_features = {'img' : tf.io.FixedLenFeature([], tf.string),
                            'eye' : tf.io.FixedLenFeature([], tf.string)}

        # Load one example
        parsed_features = tf.io.parse_single_example(proto, keys_to_features)
        # Turn your saved image string into an array (decoding)
        img = tf.image.decode_png(parsed_features['img'],channels=1) #### decode as grayscale
        eye = tf.io.decode_raw(parsed_features['eye'],out_type = tf.float32)

        ###### input preprocessing (resizing and pixel value normalization according to pretrained model requirements)
        img = tf.image.resize(img,input_shape[0:2])
        
        rotAngle = tf.random.uniform(shape = [1],minval = -90,maxval = 90,dtype = tf.int32)[0]
        heatmap = tf.numpy_function(eye_to_heatmap,[input_shape,eye],tf.uint8)[...,0]
        img,heatmap = tf.py_function(random_crop_and_resize,[img,heatmap,0.3,0.5],[tf.float32,tf.uint8])
        img = tf.numpy_function(rotate_image,[img,rotAngle,3],tf.float32)
        heatmap = tf.numpy_function(rotate_image,[heatmap,rotAngle,0],tf.uint8)

        img = tf.cast(img,dtype= tf.float32)
        heatmap = tf.cast(heatmap,dtype= tf.float32)
        heatmap = tf.stack([ heatmap[...,0], heatmap[...,1], tf.abs(1 - heatmap[...,0] - heatmap[...,1])],axis = 2)

        img = (img - tf.reduce_mean(img)) / tf.math.reduce_std(img)
        return img,heatmap

    tfrFilePath = tf.io.gfile.glob(os.path.join(tfrecordDir,'*.tfr'))
    dataset = tf.data.TFRecordDataset(tfrFilePath)

    dataset = dataset.map(_parse_function, num_parallel_calls=cpu_count())

    # Set the number of datapoints you want to load and shuffle
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    dataset = dataset.batch(batch_size = batch_size)

    return dataset

def draw_eyes(imgs_tensor,eyesAnnot,point_radius_px = 5):

    disp = tf.image.grayscale_to_rgb(imgs_tensor).numpy()
    eyesAnnot = eyesAnnot.numpy()
    # pred_entries_array[:,0] = np.round(pred_entries_array[:,0])

    b,r,c,_ = disp.shape
    for i in range(b):
        lx,ly,rx,ry = eyesAnnot[i]

        lx = int(lx * c)
        ly = int(ly * r)
        rx = int(rx * c)
        ry = int(ry * r)

        disp[i] = normalization(disp[i],0,255)
        disp[i] = cv2.circle(disp[i],center = (lx,ly),radius = point_radius_px,color = (0,0,255),thickness = -1)
        disp[i] = cv2.circle(disp[i],center = (rx,ry),radius = point_radius_px,color = (255,0,0),thickness = -1)

    disp = disp / 255
    return tf.convert_to_tensor(disp)

def scripts_backup(bckp_dir):
    if not os.path.isdir(bckp_dir):
        os.makedirs(bckp_dir)
    ############## copy all scripts to the output directory
    modelPath = scriptPath()
    thisfile = os.path.realpath(__file__)
    copy(modelPath,bckp_dir)
    copy(thisfile,bckp_dir)

if __name__ == '__main__':

    cwd = os.path.dirname(os.path.realpath(__file__))

    input_shape = [400,400,1]
    batch_size = 20
    epochs = 50
    trainTFRdir = cwd + '/tfrecords/train'
    valTFRdir = cwd + '/tfrecords/val'
    outDir = cwd + '/models/exp_2'
    trainMode = 'start'
    initLR = 0.001

    dataDir = os.path.dirname(trainTFRdir)
    scripts_backup(bckp_dir = os.path.join(outDir,'bckp'))

    if not os.path.isdir(outDir):
        os.makedirs(outDir)
    bestModelPath = os.path.join(outDir,'bestModel.h5')
    lastModelPath = os.path.join(outDir,'lastModel.h5')
    trainInfoPath = os.path.join(outDir,'cfg.json')

    ####### CREATE/LOAD TRAINING INFO JSON
    if trainMode.lower() == 'start':
        print('\nSTARTING TRAINING...')
        trainCFG = dict()
        init_epoch = 0
        trainCFG['trainTFRdir'] =  trainTFRdir
        trainCFG['valTFRdir'] = valTFRdir
        trainCFG['input_shape'] = input_shape #[int(x) for x in args.inputShape.split(',')]
        trainCFG['batch_size'] = batch_size
        trainCFG['train_epochs'] = epochs
        trainCFG['initLR'] = initLR
        trainCFG['best_val_loss'] = np.inf

        ##### INITIALIZE MODEL
        model = UNet(input_shape = input_shape)
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=initLR),loss = tf.losses.CategoricalCrossentropy())

    elif trainMode.lower() == 'resume':
        print('RESUMING TRAINING FROM: ' + trainInfoPath)
        trainCFG = load_json(trainInfoPath)
        init_epoch = trainCFG['last_epoch'] + 1
        if init_epoch >= trainCFG['train_epochs']:
            raise Exception('\nInitial training epoch value is higher than the max. number of training epochs specified')

        model = tf.keras.models.load_model(lastModelPath)

    dataset_info = load_json( os.path.join(dataDir,'data_info.json'))
    num_train_examples = dataset_info['num_examples_train']
    train_steps_per_epoch = num_train_examples // batch_size
    num_val_examples = dataset_info['num_examples_val']
    val_steps_per_epoch = num_val_examples // batch_size

    ############################## CREATE TENSORBOARD SUMMARY FILE
    currTime = get_current_time()
    summariesDir = os.path.join(outDir, 'tensorboard', currTime)
    summary_writer = tf.summary.create_file_writer(summariesDir,flush_millis = 10000)
    print('\n\nTensorboard command:\n\ntensorboard --port 3468 --logdir=%s\n\n' % summariesDir)

    for ep in range(init_epoch,epochs):
        model.optimizer.learning_rate = lr_decay_schedule(init_lr = initLR,epoch = ep,decay_factor = 1)

        ####### TRAIN LOOP
        ep_train_mean_loss = tf.metrics.Mean('train_mean_loss')

            ####### initialize the training dataset
        train_dataset = load_dataset(trainTFRdir,input_shape = input_shape,batch_size = batch_size,shuffle_buffer_size = 1000)
        startTime = time()
        for n,(imgs,heatmaps) in enumerate(train_dataset):
            batch_loss = model.train_on_batch(x = imgs,y = heatmaps)

            if np.isnan(batch_loss) or np.isinf(batch_loss):  ###### Terminate training on NaN or Inf loss value
                raise Exception('\nInvalid loss value encountered. Stopping training!!')

            ep_train_mean_loss(batch_loss)  ##### update loss mean
            time_left = get_time_left(startTime,n,train_steps_per_epoch)

            write_summary(summary_writer,batch_loss,summaryName = 'train_loss',summaryType = 'scalar',step = model.optimizer.iterations)

            if (n % 10) == 0:
                pred_heatmaps = model(imgs)
                pred_eyes = pred_heatmaps[...,0] + pred_heatmaps[...,1]
                gt_eyes = heatmaps[...,0] + heatmaps[...,1]
                write_summary(summary_writer,imgs,summaryName = 'train_imgs',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)
                write_summary(summary_writer,tf.expand_dims(gt_eyes,axis=3),summaryName = 'train_gt_eyes',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)
                write_summary(summary_writer,tf.expand_dims(pred_eyes,axis=3),summaryName = 'train_pred_eyes',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)

            printInPlace('Epoch: %d (training) -- Batch: %d/%d -- ETA: %s -- Mean loss: %f'
                        % ( ep,n,train_steps_per_epoch,time_left, ep_train_mean_loss.result()))


        write_summary(summary_writer,ep_train_mean_loss.result(),summaryName = 'train_mean_loss',summaryType = 'scalar',step = ep)

        print(' ')

        ####### VAL LOOP
        ep_val_mean_loss = tf.metrics.Mean('val_mean_loss')
            ####### initialize the validation dataset
        val_dataset = load_dataset(valTFRdir,input_shape = input_shape,batch_size = batch_size,shuffle_buffer_size = 1000)
        startTime = time()
        for n,(imgs,heatmaps) in enumerate(val_dataset):
            batch_loss = model.test_on_batch(x = imgs,y = heatmaps)
            pred_heatmaps = model(imgs,training=False)

            if np.isnan(batch_loss) or np.isinf(batch_loss):  ###### Terminate training on NaN or Inf loss value
                raise Exception('\nInvalid loss value encountered. Stopping training!!')

            ep_val_mean_loss(batch_loss)  ##### update loss mean
            time_left = get_time_left(startTime,n,val_steps_per_epoch)

            if (n % 10) == 0:
                pred_eyes = pred_heatmaps[...,0] + pred_heatmaps[...,1]
                gt_eyes = heatmaps[...,0] + heatmaps[...,1]
                write_summary(summary_writer,imgs,summaryName = 'val_imgs',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)
                write_summary(summary_writer,tf.expand_dims(gt_eyes,axis=3),summaryName = 'val_gt_eyes',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)
                write_summary(summary_writer,tf.expand_dims(pred_eyes,axis=3),summaryName = 'val_pred_eyes',summaryType = 'image',step =  model.optimizer.iterations,max_images=2)

            printInPlace('Epoch: %d (validation) -- Batch: %d/%d -- ETA: %s -- Mean loss: %f'
                        % ( ep,n,val_steps_per_epoch,time_left, ep_val_mean_loss.result()))

        write_summary(summary_writer,ep_val_mean_loss.result(),summaryName = 'val_mean_loss',summaryType = 'scalar',step = ep)

        ##### MODEL SAVING
        if ep_val_mean_loss.result() < trainCFG['best_val_loss']:
            print('\n\tMean validation loss decreased from %f to %f. \n\tSaving model to: %s' % (trainCFG['best_val_loss'],ep_val_mean_loss.result(),bestModelPath))
            model.save(filepath=bestModelPath)
            trainCFG['best_val_loss'] = float(ep_val_mean_loss.result())

        model.save(filepath = lastModelPath)

        trainCFG['last_epoch'] = ep
        save_json(trainCFG,trainInfoPath)

        print(' ')
