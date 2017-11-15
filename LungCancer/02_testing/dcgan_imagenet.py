from __future__ import print_function
# from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os
import sys
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
class BlackboxDCGAN(object):
    def __init__(self, image_size, image_channel):
        self.image_size = image_size
        self.image_channel = image_channel
        self.z_dim = 100
        self.train_images = list()
        self.train_labels = list()

    def merge(self, images, size):
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

    def read_and_decode(self, filename_queue):
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

    def saveImagePatches(self, img_out, lab_out, h=28, w=28, win=1):
        i = 1
        r2 = 0
        c2 = 0
        while True:
            j = 1
            r1 = r2 - win if i > 1 else 0
            r2 = i * h - (i - 1) * win
            # print("r1, r2", r1, r2)

            if r2 >= img_out.shape[0]:
                break

            while True:
                c1 = c2 - win if j > 1 else 0
                c2 = j * w - (j - 1) * win
                # print("r1: {0} - r2: {1} , c1: {2} - c2: {3}".format(r1, r2, c1, c2))

                if c2 >= img_out.shape[1]:
                    break

                img = img_out[r1: r2, c1: c2, :]
                print("img shape", img.shape)
                self.train_images.append(img)
                self.train_labels.append(lab_out)
                j += 1
            i += 1

    def tensor_to_image(self):
        input_path = os.path.join(FLAGS.data_dir, 'train-*')
        data_files = tf.gfile.Glob(input_path)
        # print(data_files)
        for next_slide in data_files[0:32]:
            print ("next slide: ", next_slide)
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                with tf.device("/gpu:0"):
                    print (next_slide)
                    filename_queue = tf.train.string_input_producer([next_slide])
                    image, label = self.read_and_decode(filename_queue)
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

                        # 32 size image patches from each image - window size of 9
                        # self.saveImagePatches(img_out=img_out, lab_out=lab_out)

                        self.train_images.append(img_out)
                        self.train_labels.append(lab_out)

                        # save the original images
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
                            self.saveImage(img_out, os.path.join(dir_name, str(i) + '.png'))
                    coord.request_stop()
                    coord.join(threads)
        self.train_images = np.array(self.train_images)
        self.train_labels = np.array(self.train_labels)
        print(self.train_images.shape)
        print (self.train_labels.shape)
        return self.train_images, self.train_labels

    def saveImage(self, img_out, dir_name):
        print("img_out.dtype", img_out.dtype)
        print("img_out", img_out)
        img_out *= 255.
        img_out = np.clip(img_out, 0, 255).astype('uint8')
        print("img_out", img_out)
        imsave(dir_name, img_out)

    def readTFRecord(self):
        input_path = os.path.join(FLAGS.data_dir, 'test_*')
        data_files = tf.gfile.Glob(input_path)
        # print(data_files)
        count_slides = 0
        train_images = np.empty([FLAGS.batch_size, self.image_size, self.image_size, self.image_channel],
                                dtype=np.float32)
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

    def lrelu(self,x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def generator(self, z, filters=(128, 64, 32, 16, 3), init_width=19, kernel_size=4, reuse=True):

        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                            reuse=reuse,
                            normalizer_fn=slim.batch_norm):
            with tf.variable_scope("gen"):
                net = z
                print ("gen net: ", net)
                # net = slim.fully_connected(net, init_width ** 2 * filters[0], scope='gen_fc1')
                net = fc(
                    input_vector=net,
                    num_output_length=filters[0] * init_width * init_width,
                    name="gen_fc1"
                )
                print ("gen fc net", net)
                net = tf.reshape(net, [-1, init_width, init_width, filters[0]])
                print ("gen fc reshped net: ", net)
                net = lrelu(net)
                print ("gen relu: ", net)
                for i in range(1, len(filters) - 3):
                    net = deconv2d(input_map=net,
                                   output_shape=[net.get_shape()[0],
                                                init_width * 2 ** (i + 1),
                                                init_width * 2 ** (i + 1),
                                                filters[i]],
                                   size_kernel=kernel_size,
                                   name="gen_deconv_"+str(i)
                                   )
                    print("gen net: {0} - {1}".format(i, net))
                    net = self.lrelu(net, name="gen_relu" + str(i))
                    print("gen net relu: ", net)
                    tf.summary.histogram('gen_deconv_'+str(i), net)

                # create the output with image channel
                i = len(filters) - 2
                net = deconv2d(input_map=net,
                               output_shape=[net.get_shape()[0],
                                             init_width * 2 ** (i + 1),
                                             init_width * 2 ** (i + 1),
                                             filters[i]],
                               size_kernel=kernel_size,
                               stride=2,
                               name="gen_deconv"+str(i))
                print("gen net: {0} - {1}".format(i, net))
                net = self.lrelu(net)
                print ("gen net relu: ", net)

                i = len(filters) - 1
                net = deconv2d(input_map=net,
                               output_shape=[net.get_shape()[0],
                                             self.image_size,
                                             self.image_size,
                                             filters[i]],
                               size_kernel=kernel_size,
                               stride=2,
                               name="gen_deconv" + str(i))
                net = tf.nn.tanh(net, name="gen_tanh")
                print ("gen net tanh: ", net)

                net = tf.image.resize_images(net, [self.image_size, self.image_size])
                print ("gen net reshape: ", net)

                tf.summary.histogram('gen/out', net)
                tf.summary.image("gen", net, max_outputs=8)
        return net

    def discriminator(self, x, name, filters=(16, 32, 64, 128, 256), kernel_size=5,
                      classification=False, dropout=None, int_feats=False):
        tf.summary.histogram(name, x)
        tf.summary.image(name, x, max_outputs=8)
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=lrelu):
            with tf.variable_scope(name):
                net = x
                print ("net conv: ", net)
                net = tf.pad(net, paddings=[[0, 0], [2, 2], [2, 2], [0, 0]], mode="SYMMETRIC", name="dis_padding")
                print ("net pad: ", net)
                for i in range(len(filters) - 2):
                    net = conv2d(net,
                                 num_output_channels=filters[i],
                                 size_kernel=kernel_size,
                                 name="dis_conv_"+str(i))
                    print ("net conv: {0} - {1}".format(i, net))
                    net = self.lrelu(net, name="dis_relu"+str(i))
                    print ("net relu: ", net)
                    tf.summary.histogram('dis_conv_'+str(i), net)

                i = len(filters) - 1
                net = conv2d(net,
                             num_output_channels=filters[-1],
                             size_kernel=kernel_size,
                             name="dis_conv_"+str(i))
                print ("net conv {0} - {1}".format(i, net))
                net = self.lrelu(net, name="dis_relu_"+str(i))
                print ("net relu: ", net)
                tf.summary.histogram('dis_conv_'+str(i), net)

                net = tf.contrib.layers.flatten(net, scope="dis_flatten")
                # net = tf.reshape(net, [net.get_shape()[0]._value,
                                       # net.get_shape()[1]._value * net.get_shape()[2]._value * net.get_shape()[3]])
                print ("flatten: ", net)

                net = fc(input_vector=net, num_output_length=1024, name="disc_fc_1")
                print ("fc1: ", net)
                tf.summary.histogram('dis_fc1', net)

                net = tf.nn.dropout(net, keep_prob=0.5, name="dis_dropout")
                print ("dropout: ", net)

                net = fc(input_vector=net, num_output_length=1, name="disc_fc_2")
                print ("fc2: ", net)
                tf.summary.histogram('dis_fc2', net)
        return net

    #############
    # DC-GAN #
    #############

    def mnist_gan(self, train_images, train_labels):
        # Models
        print ("Model Training")
        x = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, self.image_channel], name='X')
        # axis = list(range(len(inputs.get_shape()) - 1))
        # mean, variance = tf.nn.moments(inputs, axis, name="mean_var")
        # params_shape = inputs[-1:]
        # print ("params shape: ", params_shape)
        # beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer(),
        #                        trainable=True, dtype=tf.float32)
        # gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer(),
        #                         trainable=True, dtype=tf.float32)
        # x = tf.nn.batch_normalization(inputs, mean=mean, variance=variance, offset=beta, scale=gamma,
        #                               variance_epsilon=0.001, name="Batch_norm")
        # x.set_shape(inputs.get_shape())
        z = tf.placeholder(tf.float32, shape=[None, self.z_dim], name='z')
        print ("Building models!0")
        d_model = self.discriminator(x, name="disc1_1")
        print ("d_model: ", d_model)

        g_model = self.generator(z, reuse=False)
        print ("g_model: ", g_model)

        dg_model = self.discriminator(g_model, name="disc2")
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
                    # z_batch = np.random.uniform(-1, 1, [FLAGS.batch_size, z_dim]).astype(np.float32)
                    # trying normal distribution than normal
                    z_batch = np.random.normal(loc=0.0, scale=1.0, size=[FLAGS.batch_size, self.z_dim]).astype(np.float32)
                    idx = np.random.randint(len(train_images), size=FLAGS.batch_size)
                    print ("len idx: ", len(idx))
                    images = train_images[idx, :, :, :]
                    print ("images shape: ", images.shape)
                    x_min = np.min(images, axis=tuple(range(images.ndim - 1)), keepdims=True)
                    x_max = np.max(images, axis=tuple(range(images.ndim - 1)), keepdims=True)
                    # normalize the image array between -1 to 1
                    images = 2 * (images - x_max) / -(x_max - x_min) - 1
                    print ("images checks for -1 to 1", images.shape)

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
                    sys.stdout.flush()
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
                            # z2 = np.random.uniform(-1.0, 1.0, size=[samples, z_dim]).astype(np.float32)
                            # trying normal distribution other than uniform
                            z2 = np.random.normal(loc=0.0, scale=1.0, size=[samples, self.z_dim]).astype(np.float32)
                            print ("z2 image shape: ", z2.shape)
                            images = sess.run(g_model, feed_dict={z: z2})
                            print ("sample image shape: ", images.shape)
                            images = np.reshape(images, [samples, self.image_size, self.image_size, self.image_channel])
                            images = (images + 1.) / 2.
                            print ("random comp: ", images)
                            images = (255 * (images - np.max(images)) / -np.ptp(images)).astype(int)
                            scipy.misc.imsave(FLAGS.sampledir + '/sample'+str(step)+'.png',
                                              self.merge(images, [int(math.sqrt(samples))] * 2))
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


    def gan_class(self, train_images, train_labels):
        # Models
        dropout = .5
        x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        y = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        c_model = self.discriminator(
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

    def kmeans(self, train_images, train_labels):
        # Models
        x = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        feat_model = self.discriminator(x, name="disc", int_feats=True)
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
                    scipy.misc.imsave('/tmp/cluster%s.png' % i, self.merge(images_, [int(math.sqrt(samples))] * 2))
                return


    ##########
    # Sample #
    ##########
    def sample(self):
        if not FLAGS.sampledir:
            print(FLAGS.sampledir, 'is not defined')
            return

        # Model
        z_dim = 100
        z = tf.placeholder(tf.float32, shape=[None, z_dim])
        g_model = self.generator(z, reuse=False)
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
                scipy.misc.imsave(output, self.merge(images, [int(math.sqrt(samples))] * 2))


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
    parser.add_argument('--image_save', default=False, action='store_false',
                        help="True if require to save the image")
    parser.add_argument('--imagesavedir', type=str,
                        default=os.path.join(BASE_DIR, 'pathology', 'imagesavedir'))
    parser.add_argument('--subset', type=str, default='train')
    FLAGS, unparsed = parser.parse_known_args()
    bl = BlackboxDCGAN(image_size=299, image_channel=3)
    images, labels = bl.tensor_to_image()
    print ("loaded dataset!")
    bl.mnist_gan(images, labels)