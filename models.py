import tensorflow as tf
import numpy as np
import cnnf


class Clf:
    def __init__(self, args):
        # self.args = args
        self.in_images = tf.placeholder(
            "float32", [None, 224, 224, 3], name="input_images")
        self.in_labels = tf.placeholder(
            "float32", [None, args.n_class], name="input_labels")
		self.training = tf.placeholder("bool", [], name="training")

        self.fc7, self.logit, self.pslab, self.accuracy = self._forward(args)
        self.loss_clf, self.loss_reg = self._add_loss(args)
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        # self.lr = tf.train.exponential_decay(args.base_lr,
        #                                      self.global_step,
        #                                      args.decay_step,
        #                                      args.decay_rate,
        #                                      staircase=True,
        #                                      name="learning_rate")
        # self.optimizer = tf.train.MomentumOptimizer(self.lr, args.momentum)
        self.optimizer = tf.train.AdamOptimizer(args.base_lr,
                                                beta1=args.momentum)
        self.train_op = self.optimizer.minimize(self.loss_clf + self.loss_reg)
        self.merged = tf.summary.merge_all()

    def _forward(self, args):
        fc7 = cnnf.CNN_F(self.in_images, args.cnnf_weight, self.training)
        dim_fc7 = fc7.shape.as_list()[-1]

        # pseudo label branch
        logit = _fc(fc7, dim_fc7, args.n_class, "logit")
        pslab = tf.nn.softmax(logit, name="pseudo_label")

        # evaluate
        acc = tf.equal(tf.argmax(self.in_labels, axis=1),
                       tf.argmax(pslab, axis=1))
        acc = tf.reduce_mean(tf.to_float(acc))
        tf.summary.scalar("accuracy", acc)

        return fc7, logit, pslab, acc

    def _add_loss(self, args):
        # cross entropy
        with tf.name_scope("xent"):
            loss_xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.in_labels,
                                                                logits=self.logit)
            loss_xent = tf.reduce_sum(loss_xent, axis=-1)
            loss_clf = tf.reduce_mean(loss_xent)

        # weight decay
        var_list = [v for v in tf.trainable_variables()]
        with tf.name_scope("weight_decay"):
            loss_reg = args.weight_decay * tf.reduce_mean([tf.nn.l2_loss(x)
                                                           for x in var_list if 'weight' in x.name])

        tf.summary.scalar("loss_clf", loss_clf)
        tf.summary.scalar("loss_reg", loss_reg)
        return loss_clf, loss_reg

    def train_one_step(self, sess, images, labels):
        _, summary, l_clf, l_reg, acc = sess.run(
            [self.train_op, self.merged,
                self.loss_clf, self.loss_reg, self.accuracy],
            feed_dict={self.in_images: images,
                       self.in_labels: labels,
					   self.training: True})
        return summary, l_clf, l_reg, acc


"""
helper functions
"""


def _fc(x, num_in, num_out, var_scope, relu=False, stddev=0.01):
    """fully connected layer"""
    with tf.variable_scope(var_scope) as scope:
        weights = tf.get_variable('weight', initializer=tf.truncated_normal(
            [num_in, num_out], stddev=stddev))
        biases = tf.get_variable(
            'bias', initializer=tf.constant(0.1, shape=[num_out]))
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            relu = tf.nn.relu(act)
            return relu
        else:
            return act
