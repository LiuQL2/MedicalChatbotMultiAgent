# 修改anfis网络结构，只求重要特征的隶属度函数
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from time import time
import warnings
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
pd.set_option('max_columns',10000)
pudong=pd.read_excel('knn_result_e.xlsx')
data=pudong[['lon','lat','szlc','jzmj','fwcx',
             'jcnd','zxqk','jzjg','thbl','pbdt','jyqs',
             'shi','ting','chu','wei','sub_score','fm_score','shop_score'
             ,'stop_score','park_score','school_score']]
label=pudong['price_avg']
traindata,testdata,trainlabel,testlabel = train_test_split(data, label, test_size = 0.4)
print(data.columns)
print(traindata.shape)
print(testdata.shape)


# 修改anfis网络结构，只求重要特征的隶属度函数
class ANFIS:
    def __init__(self, n_inputs, n_inputs_2, n_rules, listname, learning_rate=1e-2):
        self.n = n_inputs
        self.n2 = n_inputs_2
        self.m = n_rules
        self.clist = listname
        self.inputs = tf.placeholder(tf.float32, shape=(None, n_inputs))  # Input
        self.inputs2 = tf.placeholder(tf.float32, shape=(None, n_inputs_2))
        self.targets = tf.placeholder(tf.float32, shape=(None,))  # Desired output
        mu = tf.get_variable("mu", [n_rules * n_inputs_2],
                             initializer=tf.random_normal_initializer(0, 1))  # Means of Gaussian MFS
        sigma = tf.get_variable("sigma", [n_rules * n_inputs_2],
                                initializer=tf.random_normal_initializer(0, 1))  # Standard deviations of Gaussian MFS
        # y = tf.get_variable("y", [1, n_rules], initializer=tf.random_normal_initializer(0, 1))  # Sequent centers



        self.rul = tf.reduce_prod(
            tf.reshape(
                tf.exp(-0.5 * tf.square(tf.subtract(tf.tile(self.inputs2, (1, n_rules)), mu)) / tf.square(sigma)),
                (-1, n_rules, n_inputs_2)), axis=2)  # Rule activations
        print('rul', self.rul)
        self.rul_normal = tf.nn.softmax(self.rul)
        print('rul_normal', self.rul_normal)
        self.p = tf.get_variable(name='weights', shape=(self.m, self.n), initializer=tf.random_normal_initializer(0, 1))
        print('weights', self.p)
        self.bias = tf.get_variable(name='bias', shape=(self.m), initializer=tf.random_normal_initializer(0, 1))
        print('bias', self.bias)
        self.params = tf.trainable_variables()
        # w=tf.reduce_sum(tf.multiply( self.inputs,self.p), axis=1)
        w=tf.matmul(self.inputs, self.p, transpose_b=True)
        print('w',w)
        temp=w+self.bias
        print('temp',temp)
        # self.hidden = tf.matmul(self.rul_normal,temp)
        self.hidden = tf.multiply(self.rul_normal, temp)
        print('hidden', self.hidden)

        # self.out = tf.nn.sigmoid(self.hidden)
        self.out = tf.reduce_sum(self.hidden, axis=1)

        self.loss = tf.losses.huber_loss(self.targets, self.out)  # Loss function computation
        # Other loss functions for regression, uncomment to try them:
        # loss = tf.sqrt(tf.losses.mean_squared_error(target, out))
        # loss = tf.losses.absolute_difference(target, out)
        self.optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)  # Optimization step
        # Other optimizers, uncomment to try them:
        # self.optimize = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(self.loss)
        # self.optimize = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.init_variables = tf.global_variables_initializer()  # Variable initializer

    def infer(self, sess, x, targets=None):
        if targets is None:
            return sess.run(self.out, feed_dict={self.inputs: x, self.inputs2: x[self.clist]})
        else:
            return sess.run([self.out, self.loss],
                            feed_dict={self.inputs: x, self.inputs2: x[self.clist], self.targets: targets})

    def train(self, sess, x, targets):
        yp, l, _ = sess.run([self.out, self.loss, self.optimize],
                            feed_dict={self.inputs: x, self.inputs2: x[self.clist], self.targets: targets})
        return l, yp

    def plotmfs(self, sess):
        mus = sess.run(self.params[0])
        mus = np.reshape(mus, (self.m, self.n2))
        sigmas = sess.run(self.params[1])
        sigmas = np.reshape(sigmas, (self.m, self.n2))
        xn = np.linspace(0, 1.5, 1000)
        for r in range(self.m):
            if r % 4 == 0:
                plt.figure(figsize=(5, 3), dpi=80)
            for i in range(self.n2):
                if i == 0:
                    plt.plot(xn, np.exp(-0.5 * ((xn - mus[r, i]) ** 2) / (sigmas[r, i] ** 2)))
                    print('mu:{},sigma:{}'.format(mus[r, i], sigmas[r, i]))
        plt.show()
        # plt.savefig('rule{}.png'.format(r+1))


# 多次实验
for t in range(1):
    for m in range(4,5):
        print('实验次数:{},规则数量:{}'.format(t + 1, m))
        tf.reset_default_graph()
        trnData = traindata
        trnLbls = trainlabel
        chkData = testdata
        chkLbls = testlabel
        batch_size = 256

        train_data = []
        train_label = []
        data_num = trnData.shape[0]

        batch_num = int(data_num / batch_size)
        for i in range(batch_num - 1):
            train_data.append(trnData[i * batch_size:(i + 1) * batch_size])
            train_label.append(trnLbls[i * batch_size:(i + 1) * batch_size])
        print(trnData.shape, trnLbls.shape)

        alpha = 0.001  # learning rate
        # Training
        num_epochs = 600
        D = data.shape[1]
        # Initialize session to make computations on the Tensorflow graph
        with tf.device('cpu:0'):
            lname = ['lon', 'lat', 'sub_score']
            fis = ANFIS(n_inputs=D, n_inputs_2=len(lname), listname=lname, n_rules=m, learning_rate=alpha)
            with tf.Session() as sess:
                # Initialize model parameters
                sess.run(fis.init_variables)
                trn_costs = []
                val_costs = []
                time_start = time.time()
                for epoch in range(num_epochs):

                    for index in range(len(train_data)):
                        train_batch = train_data[index]
                        train_batch_lable = train_label[index]
                        trn_loss, trn_pred = fis.train(sess, train_batch, train_batch_lable)
                    val_pred, val_loss = fis.infer(sess, chkData, chkLbls)
                    if epoch == num_epochs - 1:
                        time_end = time.time()
                        print("Elapsed time: %f" % (time_end - time_start))
                        print("Validation loss: %f" % val_loss)
                        # Plot real vs. predicted
                        # pred = np.vstack((np.expand_dims(trn_pred, 1), np.expand_dims(val_pred, 1)))
                        # plt.figure(1)
                        # plt.plot(mg_series)
                        # plt.plot(pred)
                    trn_costs.append(trn_loss)
                    val_costs.append(val_loss)

                print('rule:{}'.format(m))
                val_pred, val_loss = fis.infer(sess, chkData, chkLbls)
                price_min = 9.453443401029997
                price_max = 11.887498282135859
                lm = pd.DataFrame({'pred': 2.718281828459045 ** (val_pred * (price_max - price_min) + price_min),
                                   'price_avg': 2.718281828459045 ** (chkLbls * (price_max - price_min) + price_min)})
                lm.mape = np.abs(lm.pred - lm.price_avg) / lm.price_avg
                print('rule:{}'.format(m))
                print(lm.mape.loc[lm.mape <= 0.05].count() / lm.shape[0] * 100)
                print(lm.mape.loc[(lm.mape > 0.05) & (lm.mape <= 0.1)].count() / lm.shape[0] * 100)
                print(lm.mape.loc[(lm.mape > 0.1) & (lm.mape <= 0.15)].count() / lm.shape[0] * 100)
                print(lm.mape.loc[(lm.mape > 0.15) & (lm.mape <= 0.2)].count() / lm.shape[0] * 100)
                print(lm.mape.loc[(lm.mape > 0.2) & (lm.mape <= 0.25)].count() / lm.shape[0] * 100)
                print(lm.mape.loc[lm.mape > 0.25].count() / lm.shape[0] * 100)
                print(lm.mape.mean() * 100)
                fis.plotmfs(sess)
                plt.show()
                # lm.to_excel('anfis-rule{}-epochs{}-lnrate{}.xlsx'.format(m,num_epochs,alpha),index=False)
