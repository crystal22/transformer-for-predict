# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import model as models
import hparams
import data_pre
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class DeepFM(object):
    """
    Deep FM with FTRL optimization
    """
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        self.k = config['k']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        # num of features
        self.p = feature_length
        # num of fields
        self.field_cnt = field_cnt

    def add_placeholders(self):
        self.X = tf.placeholder('float32', [None, self.p])             #input one-hot
        self.y = tf.placeholder('float32', [None,1])                     #label
        # index of none-zero features
        self.feature_inds = tf.placeholder('int32', [None,field_cnt])  #idx
        self.feature=tf.placeholder('float32',[None, field_cnt])       #features


    def inference(self):
        """
        forward propagation
        :return: labels for each sample
        """
        v = tf.Variable(tf.truncated_normal(shape=[self.p, self.k], mean=0, stddev=0.01),dtype='float32')

        # three-hidden-layer neural network, network shape of (200-200-200)
        with tf.variable_scope('DNN',reuse=False):
            # embedding layer
            y_embedding_input = tf.gather(v, self.feature_inds)
            Model = models.Transformer(hparams.Hparams().parser.parse_args())
            result=Model.result(y_embedding_input)
            # first hidden layer
        # add FM output and DNN output
        self.y_out =result
        self.y_out_prob = tf.sigmoid(self.y_out)

    def add_loss(self):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_auc(self):
        # accuracy
        ys=tf.cast(self.y,dtype=tf.bool)

        self.auc_value, self.auc_op = tf.metrics.auc(labels=ys, predictions=self.y_out_prob)
        # add summary to accuracy
        tf.summary.scalar('uc', self.auc_value)

    def train(self):
        # Applies exponential decay to learning rate
        self.global_step = tf.Variable(0, trainable=False)
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_auc()
        self.train()

def train_model(sess, model, saver, epochs=10, print_every=500):
    """training model"""
    max_auc = 0.0
    #sess.run(auc_op)
    # The training process
    step=1
    for e in range(epochs):
        for sparse_ones,labels,feature_index,features in data_pre.train_next(model.batch_size):
            print(sparse_ones[0])
            los, _, AUC, _ = sess.run([model.loss, model.train_op, model.auc_value, model.auc_op],
                                      feed_dict={model.X: sparse_ones, model.y: labels,
                                                 model.feature_inds:feature_index,model.feature:features})
            print("The step is : "+str(step))
            step+=1
            if AUC > max_auc:
                max_auc = AUC
                print('In the training set, the AUC is : ' + str(AUC))
                saver.save(sess,'ckpt/byte.ckpt',global_step=e+1)
            print('The loss is : ' + str(los))


def test_model(sess, model):
    """training model"""
    # get testing data, iterable


if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # seting fields
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train',help='train')
    args = parser.parse_args()
    mode = args.mode
    # initialize the model
    config = {}
    config['lr'] = 0.001
    config['batch_size'] = 64
    config['reg_l1'] = 2e-3
    config['reg_l2'] = 0
    config['k'] = 128
    # get feature length
    feature_length = data_pre.feature_length
    # num of fields
    field_cnt = data_pre.feature_num

    model = DeepFM(config)
    # build graph for model
    model.build_graph()

    saver = tf.train.Saver(max_to_keep=1)


    with tf.Session() as sess:
        # TODO: with every epoches, print training accuracy and validation accuracy
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # restore trained parameters
        if mode == 'train':
            print('start training...')
            train_model(sess, model, saver, epochs=20, print_every=500)
        if mode == 'test':
            print('start testing...')
            test_model(sess, model)