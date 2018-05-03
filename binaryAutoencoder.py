import tensorflow as tf
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

class BinaryAutoEncoder(object) :

    DATASET_TYPE_TRAIN = 0
    DATASET_TYPE_TEST = 1

    OTHER_TYPE = 0
    CURRENT_TYPE = 1

    def __init__(self,batch_size,num_inputs,learning_rate,activation,num_units,iterations):
        self.batch_size = batch_size
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.num_units = num_units
        self.iterations = iterations


    def build_model(self):

        #autoencoder
        self.x_autoencoder = tf.placeholder(dtype=tf.float32,shape=[None,self.num_inputs])
        self.y_autoencoder = tf.placeholder(dtype=tf.float32,shape=[None,self.num_inputs])

        self.hidden_layer_autoencoder = tf.layers.dense(inputs=self.x_autoencoder, units=self.num_units,activation=tf.nn.relu)
        self.output_layer_autoencoder = tf.layers.dense(inputs=self.hidden_layer_autoencoder,units=self.num_inputs,activation=tf.nn.relu)

        self.loss_autoencoder = tf.reduce_mean(tf.squared_difference(self.output_layer_autoencoder, self.y_autoencoder))
        optimizer_autoencoder = tf.train.AdamOptimizer(self.learning_rate)
        self.autoencoder_model = optimizer_autoencoder.minimize(self.loss_autoencoder)

        #classifier
        self.x_classification = tf.placeholder(dtype=tf.float32, shape=[None, self.num_units])
        self.class_target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.hidden_classification = tf.layers.dense(inputs=self.x_classification, units=int(self.num_units/2),activation=tf.nn.relu)

        self.hidden_classification2 = tf.layers.dense(inputs=self.hidden_classification, units=int(self.num_units/2),activation=tf.nn.relu)

        self.dropout_classification = tf.nn.dropout(self.hidden_classification2,keep_prob=0.5)
        self.classification_layer = tf.layers.dense(inputs=self.dropout_classification,units=2,activation=tf.nn.relu)

        loss_classification = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.classification_layer,labels=self.class_target))

        optimizer_classification = tf.train.AdamOptimizer(self.learning_rate)
        self.classification_model = optimizer_classification.minimize(loss_classification)

    def read_data(self,filename,test_percent):
        scaler = MinMaxScaler()
        data = pd.read_csv(filename,delimiter=',')
        data = data.replace('?',0)
        data = shuffle(data)
        self.train = data.head(int(len(data) * (1 - test_percent)))
        self.train= shuffle(self.train)
        train_except_classes = self.train.loc[:,self.train.columns != '0']
        train_except_classes = scaler.fit_transform(train_except_classes)
        train_classes = self.train.loc[:,self.train.columns == '0'].values
        self.train = np.hstack((train_except_classes,train_classes))

        self.train = [self.train[self.train[:,-1] == 1],self.train[self.train[:,-1] == 0]]

        self.test = data.tail(int(len(data) * test_percent))
        self.test = shuffle(self.test)
        test_except_classes = self.test.loc[:, self.test.columns != '0']
        test_except_classes = scaler.transform(test_except_classes)
        test_classes = self.test.loc[:, self.test.columns == '0'].values
        self.test = np.hstack((test_except_classes, test_classes))

        self.test = [self.test[self.test[:,-1] == 1], self.test[self.test[:,-1] == 0]]


    def next_batch(self,dataset_type,batch_size=None):
        if batch_size is None :
            batch_size = self.batch_size

        data = self.test
        if dataset_type == self.DATASET_TYPE_TRAIN :
            data = self.train
        elif dataset_type == self.DATASET_TYPE_TEST :
            data = self.test

        if min(len(data[0]),len(data[1])) < int(batch_size/2) :
            batch_size = min(len(data[0]),len(data[1]))

        batch_size_0 = int(batch_size / 2)
        batch_size_1 = int(batch_size / 2)
        if (batch_size) % 2 != 0:
            batch_size_0 = int(batch_size / 2) + 1
            batch_size_1 = int(batch_size / 2)

        index = random.randint(0,len(data[0]) - batch_size_0)
        batch_0 = data[0][index : index + batch_size_0].reshape([batch_size_0,self.num_inputs + 1])
        target = batch_0[:,-1]
        target_binary = []
        for i in range(len(target)) :
            if target[i] == self.OTHER_TYPE :
                binary = [1,0]
            else :
                binary = [0,1]
            target_binary.append(binary)

        index = random.randint(0, len(data[0]) - batch_size_1)
        batch_1 = data[1][index: index + batch_size_1].reshape([batch_size_1, self.num_inputs + 1])
        target = batch_1[:, -1]

        for i in range(len(target)):
            if target[i] == self.OTHER_TYPE:
                binary = [1, 0]
            else:
                binary = [0, 1]
            target_binary.append(binary)

        batch = np.vstack((batch_0,batch_1))

        return batch[:,0:self.num_inputs].reshape([batch_size,self.num_inputs]),target_binary

    def train_model(self):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        print('Training autoencoder....')
        for i in range(int(self.iterations/2)) :
            batch_x,batch_y = self.next_batch(self.DATASET_TYPE_TRAIN)
            self.sess.run(self.autoencoder_model,feed_dict={self.x_autoencoder : batch_x,self.y_autoencoder : batch_x})
            if i % 100 == 0:
                MSE = self.loss_autoencoder.eval(session=self.sess,feed_dict={self.x_autoencoder : batch_x,self.y_autoencoder : batch_x})
                print(i,'\tMSE : ',MSE)

        print('Training classifier....')
        for i in range(int(self.iterations/2)):
            batch_x, batch_y = self.next_batch(self.DATASET_TYPE_TRAIN)
            output = self.hidden_layer_autoencoder.eval(session=self.sess,feed_dict={self.x_autoencoder : batch_x})
            self.sess.run(self.classification_model, feed_dict={self.x_classification: output, self.class_target: batch_y})
            if i % 100 == 0 :
                batch_x, batch_y = self.next_batch(self.DATASET_TYPE_TEST,len(self.test[0]) + len(self.test[1]))
                output = self.hidden_layer_autoencoder.eval(session=self.sess, feed_dict={self.x_autoencoder: batch_x})
                matches = tf.equal(tf.argmax(self.classification_layer,1),tf.argmax(self.class_target,1))
                acc = tf.reduce_mean(tf.cast(matches,dtype=tf.float32))
                acc_res = self.sess.run(acc,feed_dict={self.x_classification:output,self.class_target:batch_y})
                print(i,'\tACCURACY : ',acc_res)



    def close_session(self):
        self.sess.close()