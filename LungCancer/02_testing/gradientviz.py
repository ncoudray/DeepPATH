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

def outer_grad_inception(labels,
                         batch_size=1,
                         learningrate=[0.00000099, 0.0000008, 0.0000007, 0.0000006, 0.0000005],
                         weight_decay=0.1,
                         epochs=100,
                         num_classes =1):
    checkpoint = tf.train.latest_checkpoint('/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/0_scratch')
    image = cv2.imread('/home/shaivi/Desktop/Shaivi/RA/LungCancer/pathology/test_images/20.0/12_11.jpeg')
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
        with tf.device('/gpu:0'):
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
            for lr in learningrate:
                img = image.copy()
                print lr
                for i in range(0, epochs):
                    x_grad = sess.run(var_grad[0], feed_dict={x: image, y: np.array([1])})
                    print x_grad
                    new_grad = (x_grad[0] * lr + weight_decay * lr * img)
                    print new_grad
                    if not img.dtype == "float64":
                        img = img.astype(np.float64)
                    img = np.subtract(img, new_grad)
                    # print img
                    img = (img - img.min()) / img.max()
                    # print img.shape
                    # print lr
                    if i % 10 == 0:
                        write_img(img[0], "image"+str(i))

def write_img(image, name, pause=0.016):
    plt.title(name)
    im = plt.imshow(image)
    cb = plt.colorbar(im)
    plt.draw()
    plt.pause(pause)
    cb.remove()
    if not os.path.exists("/home/shaivi/Desktop/Shaivi/RA/LungCancer/LungCancer/02_testing/img"):
        os.makedirs("/home/shaivi/Desktop/Shaivi/RA/LungCancer/LungCancer/02_testing/img")
    pdb.set_trace()
    img = tf.image.encode_png(np.clip(np.multiply(image, 255.0), 0, 255))
    img_name = "/home/shaivi/Desktop/Shaivi/RA/LungCancer/LungCancer/02_testing/img/{0}.png".format(name)
    with open(img_name, "wb+") as f:
        f.write(img.eval())

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