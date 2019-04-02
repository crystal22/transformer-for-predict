# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Transformer network
'''
import tensorflow as tf

from FM_FMM_DeepFM.transformer.modules import ff, multihead_attention, noam_scheme
class Transformer:
    '''
    xs: tuple of
        x: int32 tensor. (N, T1)
        x_seqlens: int32 tensor. (N,)
        sents1: str tensor. (N,)
    ys: tuple of
        decoder_input: int32 tensor. (N, T2)
        y: int32 tensor. (N, T2)
        y_seqlen: int32 tensor. (N, )
        sents2: str tensor. (N,)
    training: boolean.
    '''
    def __init__(self, hp):
        self.hp = hp
        self.learning_rate=0.01

    def encode(self, xs, training=True):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            x = xs

            # embedding
            enc = x # (N,  d_model)
            # enc = tf.expand_dims(enc,axis=1)
            enc *= self.hp.d_model**0.5 # scale

            enc = tf.layers.dropout(enc, self.hp.dropout_rate, training=training)

            ## Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              d_model=self.hp.d_model,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
        memory = tf.reduce_sum(enc,axis=2)

        # memory=tf.squeeze(enc)
        logits=memory

        # with tf.variable_scope('fist_full_layer',reuse=tf.AUTO_REUSE):
        #     w = tf.get_variable('weight', [self.hp.d_model, 256],
        #                         initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     bias = tf.get_variable('bias', [256],
        #                            initializer=tf.constant_initializer(0))
        #     layer = tf.nn.relu(tf.matmul(memory, w) + bias)
        # with tf.variable_scope("second_full_layer",reuse=tf.AUTO_REUSE):
        #     w=tf.get_variable('weight',[256,2],
        #                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     bias=tf.get_variable('bias',[2],
        #                      initializer=tf.constant_initializer(0))
        #     logits=tf.matmul(layer, w) + bias
        return logits
    def result(self,x,training=True):
        # with tf.variable_scope('embedding',reuse=tf.AUTO_REUSE):
        #     embedding=tf.get_variable(name='emb',shape=[x1.shape[1],self.hp.d_model],dtype=tf.float32)
        # embeddings=tf.nn.embedding_lookup(embedding,index)
        # input1=tf.multiply(embeddings,tf.reshape(x1,[-1,x1.shape[1],1]))
        out=self.encode(x,training)
        # logit2=self.encode(x2,training)
        # with tf.variable_scope('fist_full_layer1', reuse=tf.AUTO_REUSE):
        #     w = tf.get_variable('weight', [x1.shape[1], 128],
        #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     bias = tf.get_variable('bias', [128],
        #                            initializer=tf.constant_initializer(0))
        #     logit1 = (tf.matmul(x1, w) + bias)
        # with tf.variable_scope('fist_full_layer2', reuse=tf.AUTO_REUSE):
        #     w = tf.get_variable('weight', [x2.shape[1], 128],
        #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     bias = tf.get_variable('bias', [128],
        #                            initializer=tf.constant_initializer(0))
        #     logit2 = (tf.matmul(x2, w) + bias)
        #log=tf.concat([logit1,logit2],axis=1)

        # with tf.variable_scope('fist_full_layer', reuse=tf.AUTO_REUSE):
        #     w = tf.get_variable('weight', [10, 128],
        #                             initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     bias = tf.get_variable('bias', [128],
        #                            initializer=tf.constant_initializer(0))
        #     logit3 = (tf.matmul(x3, w) + bias)

        # with tf.variable_scope('embedding3',reuse=tf.AUTO_REUSE):
        #     embedding3=tf.get_variable(name='emb',shape=[x3.shape[1],self.hp.d_model],dtype=tf.float32)
        # embeddings3=tf.nn.embedding_lookup(embedding3,index3)
        # input3=tf.multiply(embeddings3,tf.reshape(x3,[-1,x3.shape[1],1]))
        # logit3=self.encode(input3,training)

        # concate=tf.concat([logit1,logit2,logit3],axis=1)
        #concate=logit3
        with tf.variable_scope("second_full_layer",reuse=tf.AUTO_REUSE):
            w=tf.get_variable('weight',[11,1],
                              initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias=tf.get_variable('bias',[1],
                             initializer=tf.constant_initializer(0))
            logits=tf.matmul(out, w) + bias
        return logits
    #def train(self, xs, ys):
    def train(self,x, ys):
        '''
        Returns
        loss: scalar.
        train_op: training operation
        global_step: scalar.
        summaries: training summary node
        '''


        # forward
        #logits= self.encode(xs)

        logits=self.result(x)
        # predicts=tf.nn.softmax(logits)
        predicts = tf.sigmoid(logits)

        # cross_entropy = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(predicts, 1e-10, 1.0)))
        #ys=tf.one_hot(ys, depth=2)
        ##loss function
        sigmoid_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ys)
        sigmoid_loss = tf.reduce_mean(sigmoid_loss)
        cross_entropy = sigmoid_loss

        loss=cross_entropy
        # train scheme
        # y_ = label_smoothing(tf.one_hot(ys, depth=self.hp.vocab_size))
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=memory, labels=y_)

        # global_step = tf.train.get_or_create_global_step()
        # lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        #tf.summary.scalar('lr', lr)
        tf.summary.scalar("loss", loss)
        #tf.summary.scalar("global_step", global_step)

        #summaries = tf.summary.merge_all()
        ys=tf.cast(ys,dtype=tf.bool)

        auc_value, auc_op = tf.metrics.auc(labels=ys, predictions=predicts)

        return loss, train_op, auc_value, auc_op, predicts[0]

    #def eval(self, xs):
    def eval(self,x):
        '''Predicts autoregressively
        At inference, input ys is ignored.
        Returns
        y_hat: (N, T2)
        '''
        #ys=tf.one_hot(ys, depth=self.hp.vocab_size)
        #logits= self.encode(xs, False)
        logits = self.result(x,False)
        #predicts = tf.sigmoid(logits)
        logits=tf.nn.softmax(logits)
        predicts=logits
        # correct_prediction = tf.equal(tf.arg_max(memory, 1), tf.arg_max(ys, 1))
        # AUC = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #auc_value, auc_op=tf.metrics.auc(labels=ys,predictions=predicts,num_thresholds=2000)
        return predicts