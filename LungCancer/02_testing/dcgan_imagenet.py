from __future__ import print_function
# from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os
import scipy.misc
import tensorflow as tf
from sklearn.cluster import KMeans
from inception.nc_imagenet_data import ImagenetData
from inception.image_processing import inputs
import random
from dcgan_ops import *

slim = tf.contrib.slim
tf.set_random_seed(1)
np.random.seed(1)
tf.logging.set_verbosity(tf.logging.INFO)

#########
# Model #
#########
def merge(images, size):
    print ("size: ", size)
    h, w = images.shape[1], images.shape[2]
    print ("h: {0}, w: {1} ".format(h, w))
    img = np.zeros([h * size[0], w * size[1], images.shape[3]])
    print ("img:", img.shape)
    for idx, image in enumerate(images):
        i = int(idx % size[1])
        j = int(idx // size[1])
        print ("i: {0}, j: {1}".format(i, j))
        img[j * h : j * h + h, i * w : i * w + w, :] = image
    return img

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
     serialized_example,
     # Defaults are not specified since both keys are required.
     features = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                 default_value=''),
        'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                     default_value=-1),
        'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                    default_value=''),
        })
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize_images(image, [299, 299])
    label = tf.cast(features['image/class/label'], tf.int32)

    return image, label

def tensor_to_image():
    train_images = []
    train_labels = []
    input_path = os.path.join(FLAGS.data_dir, 'train-*')
    data_files = tf.gfile.Glob(input_path)
    # print(data_files)
    for next_slide in data_files[0:32]:
        print ("next slide: ", next_slide)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            with tf.device("/gpu:0"):
                print (next_slide)
                filename_queue = tf.train.string_input_producer([next_slide])
                image, label = read_and_decode(filename_queue)
                # print ("image shape: ", image.shape)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                init_op = tf.global_variables_initializer()
                sess.run(init_op)
                nbr_slides = 0
                for record in tf.python_io.tf_record_iterator(next_slide):
                    nbr_slides += 1

                print(nbr_slides)
                for i in range(nbr_slides):
                    img_out, lab_out = sess.run([image, label])
                    train_images.append(img_out)
                    train_labels.append(lab_out)
                    if FLAGS.image_save:
                        slide = next_slide.split("/")[-1]
                        print ("imagesavedir: ", FLAGS.imagesavedir)
                        print ("slide: ", slide)
                        dir_name = os.path.join(FLAGS.imagesavedir, slide)
                        print ("dirname: ", dir_name)
                        if not os.path.exists(dir_name):
                            print ("Inside")
                            os.makedirs(dir_name)
                            print ("Made dirname: ", dir_name)
                        saveImage(img_out, os.path.join(dir_name, str(i) + '.png'))
                coord.request_stop()
                coord.join(threads)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print(train_images.shape)
    print (train_labels.shape)
    return train_images, train_labels

def saveImage(img_out, dir_name):
    print("img_out.dtype", img_out.dtype)
    print("img_out", img_out)
    img_out /= 2.
    img_out += 0.5
    img_out *= 255.
    img_out = np.clip(img_out, 0, 255).astype('uint8')
    print("img_out", img_out)
    imsave(dir_name, img_out)

def readTFRecord():
    input_path = os.path.join(FLAGS.data_dir, 'test_*')
    data_files = tf.gfile.Glob(input_path)
    # print(data_files)
    count_slides = 0
    train_images = np.empty([FLAGS.batch_size, 299, 299, 3], dtype=np.float32)
    train_labels = np.empty([FLAGS.batch_size], dtype=np.int32)
    for i, next_slide in enumerate(data_files):
        print("New Slide ------------ %d" % (count_slides))
        print ("Slide: ", next_slide)
        labelindex = int(next_slide.split('_')[-1].split('.')[0])
        if labelindex == 1:
            labelname = 'luad'
        elif labelindex == 2:
            labelname = 'lusc'
        else:
            labelname = 'error_label_name'
        print("label %d: %s" % (labelindex, labelname))

        FLAGS.data_dir = next_slide
        dataset = ImagenetData(subset=FLAGS.subset)
        print (dataset.num_classes())

        images, labels, all_filenames, filename_queue = inputs(dataset)

        print (images)
        print (labels)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            example, lab = sess.run([images, labels])
            if i == 0:
                train_images = example
                train_labels = lab
            else:
                train_images = np.append(train_images, example, axis=0)
                train_labels = np.append(train_labels, lab, axis=0)
            print ("train: ", train_images)
            coord.request_stop()
            coord.join(threads)
    print ("len: ", train_images.shape)
    print ("len L: ", train_labels.shape)
    return train_images, train_labels

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def generator(z, reuse=True):
    init_width = 19
    filters = (128, 64, 32, 16, 3)
    kernel_size = 5
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        reuse=reuse,
                        normalizer_fn=slim.batch_norm):
        with tf.variable_scope("gen"):
            net = z
            print ("gen net: ", net)
            net = slim.fully_connected(net, init_width ** 2 * filters[0], scope='gen_fc1')
            print ("gen fc net", net)
            net = tf.reshape(net, [-1, init_width, init_width, filters[0]])
            print ("gen fc reshped net: ", net)
            for i in range(1, len(filters) - 1):
                net = slim.conv2d_transpose(
                    net, filters[i],
                    kernel_size=kernel_size,
                    stride=2,
                    scope='gen_deconv_'+str(i))
                print("gen net: {0} - {1}".format(i, net))
                net = lrelu(net, name="gen_relu" + str(i))
                print("gen net relu: ", net)
                tf.summary.histogram('gen_deconv_'+str(i), net)

            i = len(filters)
            net = slim.conv2d_transpose(
                net, filters[-1],
                kernel_size=kernel_size,
                stride=1,
                scope='gen_deconv_' + str(i))
            print("gen net: {0} - {1}".format(i, net))
            net = tf.nn.tanh(net, name="gen_tanh")
            print ("gen net tanh: ", net)

            net = tf.image.resize_images(net, [299, 299])
            print ("gen net reshape: ", net)

            tf.summary.histogram('gen/out', net)
            tf.summary.image("gen", net, max_outputs=8)
    return net

def discriminator(x, name, classification=False, dropout=None, int_feats=False):
    filters = (16, 32, 64, 128, 256)
    kernel_size = 5
    tf.summary.histogram(name, x)
    tf.summary.image(name, x, max_outputs=8)
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=lrelu):
        with tf.variable_scope(name):
            net = x
            print ("net conv: ", net)
            net = tf.pad(net, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode="SYMMETRIC", name="dis_padding")
            print ("net pad: ", net)
            for i in range(len(filters) - 1):
                net = conv2d(net,
                             num_output_channels=filters[i],
                             size_kernel=kernel_size,
                             name="dis_conv_"+str(i))
                print ("net conv: {0} - {1}".format(i, net))
                net = lrelu(net, name="dis_relu"+str(i))
                print ("net relu: ", net)
                tf.summary.histogram('dis_conv_'+str(i), net)

            i = len(filters)
            net = conv2d(net,
                         num_output_channels=filters[-1],
                         size_kernel=kernel_size,
                         name="dis_conv_"+str(i))
            print ("net conv {0} - {1}".format(i, net))
            net = lrelu(net, name="dis_relu_"+str(i))
            print ("net relu: ", net)
            tf.summary.histogram('dis_conv_'+str(i), net)
            net = slim.flatten(net,)
            print ("flatten: ", net)
            net = slim.fully_connected(net, 1024, activation_fn=None, scope='dis_fc1')
            print ("fc1: ", net)
            tf.summary.histogram('dis_fc1', net)
            net = tf.nn.dropout(net, keep_prob=0.5, name="dis_dropout")
            print ("dropout: ", net)
            net = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='dis_out')
            print ("fc2: ", net)
            tf.summary.histogram('dis_fc2', net)
    return net

#############
# DC-GAN #
#############

def mnist_gan(train_images, train_labels):
    # Models
    print ("Model Training")
    z_dim = 100
    x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name='X')
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    print ("Building models!0")
    d_model = discriminator(x, name="disc1_1")
    print ("d_model: ", d_model)

    g_model = generator(z, reuse=False)
    print ("g_model: ", g_model)

    dg_model = discriminator(g_model, name="disc2")
    print ("dg_mode: ", dg_model)

    tf.add_to_collection("d_model", d_model)
    tf.add_to_collection("dg_model", dg_model)
    tf.add_to_collection('g_model', g_model)

    # Optimizers
    t_vars = tf.trainable_variables()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    print ("global_step: ", global_step)

    d_loss = -tf.reduce_mean(tf.log(d_model) + tf.log(1. - dg_model), name='d_loss')
    print ("d_loss: ", d_loss)

    tf.summary.scalar('d_loss', d_loss)

    d_trainer = tf.train.GradientDescentOptimizer(FLAGS.d_learn, name='d_adam').minimize(
        d_loss,
        global_step=global_step,
        var_list=[v for v in t_vars if 'disc' in v.name],
        name='d_min')
    print ("d_trainer: ", d_trainer)
    # tf.summary.scalar('d_trainer', d_trainer)

    g_loss = -tf.reduce_mean(tf.log(dg_model), name='g_loss')
    print ("g_loss: ", g_loss)
    tf.summary.scalar('g_loss', g_loss)
    g_trainer = tf.train.AdamOptimizer(FLAGS.g_learn, beta1=.5, name='g_adam').minimize(
        g_loss,
        var_list=[v for v in t_vars if 'gen' in v.name],
        name='g_min')
    print ("g_trainer: ", g_trainer)
    # tf.summary.scalar(g_trainer, "g_trainer")
    init = tf.global_variables_initializer()
    print ("init")
    # Session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        with tf.device('/gpu:0'):
            sess.run(init)
            saver = tf.train.Saver(max_to_keep=20)
            checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
            if checkpoint and not FLAGS.debug:
                print('Restoring from', checkpoint)
                saver.restore(sess, checkpoint)

            summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)

            # Training loop
            print ("Staring training")
            for step in range(2 if FLAGS.debug else int(1e6)):
                print ("Step: ", step)
                z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
                idx = np.random.randint(len(train_images), size=FLAGS.batch_size)
                print ("len idx: ", len(idx))
                images = train_images[idx, :, :, :]
                print ("images shape: ", images.shape)
                # normalize the image array between -1 to 1
                images = 2 * (images - np.max(images)) / -np.ptp(images) - 1
                print ("images checks for -1 to 1", images[1,299, 299, 3])

                # Update discriminator twice
                sess.run(d_trainer, feed_dict={x: images, z: z_batch})
                _, d_loss_val = sess.run([d_trainer, d_loss], feed_dict={x: images, z: z_batch})
                # Update generator
                # sess.run(g_trainer, feed_dict={z: z_batch})
                _, g_loss_val = sess.run([g_trainer, g_loss], feed_dict={z: z_batch})

                # Log details
                print("Gen Loss: ", g_loss_val, " Disc loss: ", d_loss_val)
                print (z_batch.shape)
                summary_str = sess.run(summary, feed_dict={x: images, z: z_batch})
                summary_writer.add_summary(summary_str, global_step.eval())

                # Early stopping
                if np.isnan(g_loss_val) or np.isnan(g_loss_val):
                    print('Early stopping')
                    break

                if step % 100 == 0:
                    # Save samples
                    if FLAGS.sampledir:
                        if not os.path.exists(FLAGS.sampledir):
                            os.makedirs(FLAGS.sampledir)
                        samples = FLAGS.batch_size
                        z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
                        print ("z2 image shape: ", z2.shape)
                        images = sess.run(g_model, feed_dict={z: z2})
                        print ("sample image shape: ", images.shape)
                        images = np.reshape(images, [samples, 299, 299, 3])
                        images = (images + 1.) / 2.
                        print ("random comp: ", images)
                        scipy.misc.imsave(FLAGS.sampledir + '/sample'+str(step)+'.png',
                                          merge(images, [int(math.sqrt(samples))] * 2))
                        print ("save sample images")
                    # save model
                    if not FLAGS.debug:
                        checkpoint_file = os.path.join(FLAGS.logdir, "checkpoint")
                        saver.save(sess, checkpoint_file, global_step=global_step)
                        print ("Checkpoint saved for {0} step".format(str(step)))
            return

##################
# Gan classifier #
##################


def gan_class(train_images, train_labels):
    # Models
    dropout = .5
    x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    y = tf.placeholder(tf.float32, shape=[None, 2])
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    c_model = discriminator(
        x,
        name="disc1",
        classification=True,
        dropout=keep_prob)

    # Loss
    t_vars = tf.trainable_variables()
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=c_model, labels=y, name="cross_entropy")
                          , name="reduce_mean")
    optimizer = tf.train.AdamOptimizer(1e-4, beta1=.5, name="adam_opt")
    trainer = optimizer.minimize(
        loss,
        var_list=[v for v in t_vars if 'classify/' in v.name],
        name="trainer"
    )

    # Evaluation metric
    correct_prediction = tf.equal(tf.argmax(c_model, 1), tf.argmax(y, 1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    init = tf.global_variables_initializer()

    # Saver
    restore_vars = {}
    for t_var in t_vars:
        if 'conv' in t_var.name:
            restore_vars[t_var.name.split(':')[0]] = t_var
    saver = tf.train.Saver(restore_vars, max_to_keep=20)

    # Session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        with tf.device('/gpu:0'):
            sess.run(init)
            checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
            if not checkpoint:
                return
            saver.restore(sess, checkpoint)

            # Training loop
            for i in range(2000):
                idx = np.random.randint(len(train_images), size=FLAGS.batch_size)
                images = train_images[idx, :, :, :]
                print("images shape: ", images.shape)
                # Train
                _, loss_val = sess.run([trainer, loss], feed_dict={x: images, y: labels, keep_prob: dropout})
                if i % 100 == 0:
                    print('Loss', loss_val)
                if i % 400 == 0:
                    idx_test = np.random.randint(len(train_images), size=FLAGS.batch_size)
                    test_images = train_images[idx_test,:,:,:]
                    test_labels = [idx_test]
                    test_accuracy = sess.run(accuracy, feed_dict={
                        x: test_images,
                        y: test_labels,
                        keep_prob: 1.})
                    print("test accuracy %g" % test_accuracy)
            return


#####################
# KMeans classifier #
#####################

def kmeans(train_images, train_labels):
    # Models
    x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
    feat_model = discriminator(x, name="disc", int_feats=True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        with tf.device('/gpu:0'):
            sess.run(init)
            # Restore model params
            checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
            saver.restore(sess, checkpoint)

            # Extract intermediate features
            idx = np.random.randint(len(train_images), size=FLAGS.batch_size)
            images, labels = train_images[idx,:,:,:], train_labels[idx]
            im_features = sess.run(feat_model, feed_dict={x: images})

            # Run kmeans and evaluate
            kmeans = KMeans(n_clusters=10, random_state=0).fit(im_features)
            km_labels = kmeans.labels_
            for i in range(10):
                images_ = images[np.where(km_labels == i)[0]]
                samples = FLAGS.batch_size
                images_ = np.reshape(images_[:samples], [samples, 28, 28])
                images_ = (images_ + 1.) / 2.
                scipy.misc.imsave('/tmp/cluster%s.png' % i, merge(images_, [int(math.sqrt(samples))] * 2))
            return


##########
# Sample #
##########


def sample():
    if not FLAGS.sampledir:
        print(FLAGS.sampledir, 'is not defined')
        return

    # Model
    z_dim = 100
    z = tf.placeholder(tf.float32, shape=[None, z_dim])
    g_model = generator(z, reuse=False)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Session
    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            sess.run(init)
            # Restore
            checkpoint = tf.train.latest_checkpoint(FLAGS.logdir)
            if checkpoint:
                saver.restore(sess, checkpoint)

            # Save samples
            output = FLAGS.sampledir + '/sample.png'
            samples = FLAGS.batch_size
            z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
            images = sess.run(g_model, feed_dict={z: z2})
            images = np.reshape(images, [samples, 28, 28])
            images = (images + 1.) / 2.
            scipy.misc.imsave(output, merge(images, [int(math.sqrt(samples))] * 2))


if __name__ == '__main__':
    import argparse
    FLAGS = None
    parser = argparse.ArgumentParser()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print ("BASEDIR: ", BASE_DIR)
    parser.add_argument('--logdir', type=str,
                        default=os.path.join(BASE_DIR, 'pathology', 'checkpoints'),
                        help='Directory to store Checkpoints for the model')
    parser.add_argument('--batch_size', default=10, type=int,
                        help="The size of batch images [32]")
    parser.add_argument('--d_learn', default=0.001, type=float,
                        help="Discrominator Learning rate")
    parser.add_argument('--g_learn', default=0.000002, type=float,
                        help="Generator Learning Rates")
    parser.add_argument('--data_dir', type=str,
                         default=os.path.join(BASE_DIR, 'pathology', 'test_viz'))
    parser.add_argument('--sampledir', type=str,
                        default=os.path.join(BASE_DIR, 'pathology', 'sampledir'),
                        help='Save the sample image')
    parser.add_argument('--debug', default=False, action='store_false',
                        help="True if debug mode")
    parser.add_argument('--image_save', default=True, action='store_true',
                        help="True if require to save the image")
    parser.add_argument('--imagesavedir', type=str,
                        default=os.path.join(BASE_DIR, 'pathology', 'imagesavedir'))
    parser.add_argument('--subset', type=str, default='train')
    FLAGS, unparsed = parser.parse_known_args()
    images, labels = tensor_to_image()
    print ("loaded dataset!")
    # mnist_gan(images, labels)