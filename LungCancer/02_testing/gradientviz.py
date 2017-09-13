import tensorflow as tf
from inception.slim import slim, losses
from inception import inception_model
from inception.image_processing import inputs
import matplotlib.pyplot as plt
import numpy as np
from inception.nc_imagenet_data import ImagenetData
from inception.inception_model import inference
import os
import cv2
import pdb

FLAGS = None

def outer_grad_inception(labels, batch_size=1, learningrate=0.1, weight_decay=0.1, num_classes =1):
    checkpoint = tf.train.latest_checkpoint('/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/0_scratch')
    image = cv2.imread('/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/test_images/20.0/3_21.jpeg')
    # tf_record = '/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/test_TFperSlide/test_TCGA-05-5425-01A-01-BS1.259004dc-6769-4f66-afa3-57a4d8e0edb5_1.TFRecord'
    # feature = {'image/encoded': tf.FixedLenFeature([], tf.string),
    #            'image/class/label': tf.FixedLenFeature([], tf.int64)}
    # e = tf.python_io.tf_record_iterator(tf_record).next()
    # print e
    # single_example = tf.parse_single_example(e, features=feature)
    # print single_example['image/encoded']
    image = cv2.resize(image, (299, 299),interpolation=cv2.INTER_CUBIC)
    print image.shape
    image = np.reshape(image, (1, image.shape[0], image.shape[1], 3))
    print image.shape
    x = tf.placeholder(tf.float32, [1, image.shape[1], image.shape[2], image.shape[3]])
    print x.shape
    y = tf.placeholder(tf.int32, [1])
    label = tf.one_hot(y, 2)
    logits, auxiliary_logits, endpoints, net2048 = inference(x, 2)
    with tf.get_default_graph() and tf.Session(config=tf.ConfigProto(allow_soft_placement=True))as sess:
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint))
        saver.restore(sess, checkpoint)
        print("Model restored from dir")
        print tf.global_variables()
        plt.ion()
        plt.show()
        sess.run(tf.global_variables_initializer())

        print "endpoints", endpoints
        loss_grad = slim.losses.cross_entropy_loss(logits,
                                 label,
                                 label_smoothing=0.1,
                                 weight=1.0)

        print "loss_grad", loss_grad
        var_grad = tf.gradients(loss_grad, [x])
        print "var_gard", var_grad
        print "sess var: ", sess.run(var_grad, feed_dict={x: image, y: np.array([1])})
        for i in range(10):
            x_grad = sess.run(var_grad, feed_dict={x: image, y: np.array([1])})
            print x_grad
            # image -= (x_grad[0] * learningrate + weight_decay * learningrate * image)
            # print image
            # image = (image - image.min()) / image.max()
            # print image
            # print image.shape
            # im = sess.run(image)
            # print im.shape
            # write_img(im[0])

def write_img(image, pause=0.016):
    im = plt.imshow(image)
    cb = plt.colorbar(im)
    plt.draw()
    plt.pause(pause)
    cb.remove()

def main(unused_argv=None):
    input_path = os.path.join(FLAGS.data_dir, 'test_*')
    data_files = tf.gfile.Glob(input_path)
    print(data_files)
    count_slides = 0

    for next_slide in data_files:
        print("New Slide ------------ %d" % (count_slides))
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
    print dataset

    images, labels, all_filenames, filename_queue = inputs(dataset, batch_size=1)

    print images
    print labels

    outer_grad_inception(labels)

if __name__ == '__main__':
    import numpy as np
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/test_viz',
        help='File containing abspath to the Training Data with Labels')
    parser.add_argument(
        '--subset',
        type=str,
        default='train'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

