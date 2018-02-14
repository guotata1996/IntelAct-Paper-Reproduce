import tensorflow as tf
import numpy as np
from config import *
from util import *

class Network:
    def __init__(self):
        self.global_step = tf.Variable(0, trainable=False, name='step')

        self.input_images = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.input_measurement = tf.placeholder(tf.float32, [None, num_measurements])
        self.goal = tf.placeholder(tf.float32, [None, 6, num_measurements])
        self.real_measurement = tf.placeholder(tf.float32, [None, 6, num_measurements])
        self.real_action = tf.placeholder(tf.int32, [None])
        self.time_mask = tf.placeholder(tf.float32, [None, 6])

        self.norm_input_images = self.input_images / 128.0 - 1.0
        self.per_1 = conv2d_layer(self.norm_input_images, 8, 32, 'per-1', 4)
        self.per_2 = conv2d_layer(self.per_1, 4, 64, 'per2', 2)
        self.per_3 = conv2d_layer(self.per_2, 3, 128, 'per3', 1)
        self.per_out = dense_layer(flatten(self.per_3), 1024, 'per4')

        self.mea_1 = dense_layer(self.input_measurement, 128, 'mea1')
        self.mea_2 = dense_layer(self.mea_1, 128, 'mea2')
        self.mea_out = dense_layer(self.mea_2, 128, 'mea3')

        self.flat_goal = flatten(self.goal)
        self.goal_1 = dense_layer(self.flat_goal, 128, 'goal1')
        self.goal_2 = dense_layer(self.goal_1, 128, 'goal2')
        self.goal_out = dense_layer(self.goal_2, 128, 'goal3')

        self.combined = tf.concat([self.per_out, self.mea_out, self.goal_out], 1)

        self.expectation_1 = dense_layer(self.combined, 1024, 'exp1')
        self.expectation_2 = dense_layer(self.expectation_1, num_measurements*6, 'exp2', func=None)
        self.expectation_3 = tf.reshape(self.expectation_2, [-1, 6, num_measurements])
        self.expectation_out = tf.stack([self.expectation_3]*num_actions, 2) #None x 6 x num_actions x num_measurements

        self.vantage_1 = dense_layer(self.combined, 1024, 'van1')
        self.vantage_2 = dense_layer(self.vantage_1, 6*num_measurements*num_actions, 'van2', func=None)
        self.vantage_out = tf.reshape(self.vantage_2, [-1, 6, num_actions, num_measurements])
        self.vantage_out = self.vantage_out - tf.stack([tf.reduce_mean(self.vantage_out,reduction_indices=2)]*num_actions, 2)

        self.expanded_goal = tf.stack([self.goal]*num_actions, 2) #Nonex6xnum_actionsxnum_measurements
        self.expanded_time_mask = tf.stack([self.time_mask]*num_measurements, -1) #None x 6 x num_measurements
        self.expanded_time_mask = tf.stack([self.expanded_time_mask]*num_actions, 2) #Nonex6xnum_actionsxnum_measurements
        self.objective = tf.multiply(self.vantage_out, self.expanded_time_mask)
        self.objective = tf.multiply(self.objective, self.expanded_goal) #Nonex6xnum_actionsxnum_measurements
        self.action_prediction = tf.reduce_sum(self.objective, [1, -1]) #None x num_actions

        self.delta_prediction = tf.add(self.expectation_out, self.vantage_out) #Noenx6xnum_actionsxnum_measurements
        self.expanded_input = tf.stack([self.input_measurement]*6, 1) #None x 6 x num_measurements
        self.expanded_input = tf.stack([self.expanded_input]*num_actions, 2) #None x 6 x num_actions x num_measurements
        self.measurement_prediction = tf.add(self.expanded_input, self.delta_prediction)

        self.action_mask = tf.one_hot(self.real_action, num_actions, on_value=True, off_value=False) #Nonexnum_actions
        self.action_mask = tf.stack([self.action_mask]*6, 1) #None x 6 x num_actions
        self.measurement_for_loss = tf.stack([self.real_measurement]*num_actions, 2) #Nonex6xnum_actionsxreal_measurments
        self.masked_measurement = tf.boolean_mask(self.measurement_for_loss, self.action_mask) #?x5
        self.masked_prediction = tf.boolean_mask(self.measurement_prediction, self.action_mask)  #?x5
        self.error = tf.abs(self.masked_measurement - self.masked_prediction)
        self.loss = tf.nn.l2_loss(self.masked_measurement - self.masked_prediction)
        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.95, beta2=0.999, epsilon=1e-4).minimize(self.loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.base_g = np.asarray(goal)
        self.base_g = np.stack([self.base_g]*6, 0)
        self.base_timemask = np.asarray([0, 0, 0, 0.5, 0.5, 1])

        self.summary_writer = tf.summary.FileWriter("log", self.sess.graph)
        self.summarize_loss = tf.summary.scalar('loss',self.loss)

        self.angle_error = tf.placeholder(tf.float32, shape=[])
        self.summarize_error = tf.summary.scalar('angle_err', self.angle_error)

        self.frags = tf.placeholder(tf.float32, shape=[])
        self.summarize_performance = tf.summary.scalar('Frag', self.frags)

        self.saver = tf.train.Saver()

    def predict(self, observation):
        input_images = []
        input_measurement = []
        goal = []
        time_mask = []

        for ob in observation:
            input_images.append(ob[bytes('image', encoding='utf8')])
            input_measurement.append(ob[bytes('measurement', encoding='utf8')])
            goal.append(self.base_g)
            time_mask.append(self.base_timemask)

        pred = self.sess.run(self.action_prediction, feed_dict={self.input_images:input_images, self.time_mask:time_mask,
                                                                            self.input_measurement:input_measurement, self.goal:goal})
        return pred

    def train(self, observation):
        input_images = []
        input_measurement = []
        real_action = []
        real_measurement = []
        goal = []

        for ob in observation:
            input_images.append(ob[bytes('image', encoding='utf8')])
            input_measurement.append(ob[bytes('measurement', encoding='utf8')])
            real_action.append(ob['real_action'])
            real_measurement.append(ob['real_measurement'])
            goal.append(self.base_g)

        _, loss_summary, error, step = self.sess.run([self.train_op, self.summarize_loss, self.error, self.global_step], feed_dict={self.input_images:input_images, self.input_measurement:input_measurement, self.goal:goal,
                                                self.real_measurement:real_measurement, self.real_action:real_action})
        mean_error = np.mean(error, 0)[measurement_of_interest]
        error_summary = self.sess.run(self.summarize_error, feed_dict={self.angle_error:mean_error})
        self.summary_writer.add_summary(loss_summary, step)
        self.summary_writer.add_summary(error_summary, step)
        self.summary_writer.flush()

    def _checkpoint_filename(self, episode):
        return 'checkpoints/%s_%08d' % (model_name, episode)

    def save(self):
        step = self.sess.run(self.global_step)
        self.saver.save(self.sess, self._checkpoint_filename(step))

    def restore(self):
        checkpoint_name = tf.train.latest_checkpoint('checkpoints/')
        self.saver.restore(self.sess, checkpoint_name)

    def log_performance(self, observation):
        agent = np.random.randint(len(observation))
        ob = observation[agent]
        summary = self.sess.run(self.summarize_performance, feed_dict={self.frags: ob[bytes('measurement', encoding='utf8')][0]})
        self.summary_writer.add_summary(summary)
        self.summary_writer.flush()