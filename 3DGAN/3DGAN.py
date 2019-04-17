import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import time

n_epochs = 30001
batch_size = 64
alpha_g = 0.0015
alpha_d = 0.000024
beta1 = 0.5
logs_path = "./logs"
train_sample_directory = './train_sample/'
model_directory = './models'
train_dir = './Voxel_Files/'
train_names = [filename for filename in os.listdir(train_dir) if filename.endswith('_1')]

chairs = []
for filename in train_names:
    chairs.append(np.load(train_dir+filename))
train_chairs = np.array(chairs)
train_chairs=train_chairs.reshape([989,32,32,32,1])
np.random.seed(1)
tf.reset_default_graph()
tf.set_random_seed(1)
initializer = tf.truncated_normal_initializer(stddev=2e-2)

def get_batch(batch_size):
    indices = np.random.randint(len(train_chairs), size=batch_size) # random sample real images
    batch = train_chairs[indices]
    return batch

def generator(z, batch_size=batch_size, is_train=True, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
        tl.layers.set_name_reuse(reuse)

    gen_in = tl.layers.InputLayer(inputs=z, name='gen/in')
    gen_layer0 = tl.layers.DenseLayer(layer=gen_in, n_units = 4*4*4*512, W_init = initializer, act = tf.identity, name='gen/layer0/lin')
    gen_layer0 = tl.layers.ReshapeLayer(gen_layer0, shape = [-1,4,4,4,512], name='gen/layer0/reshape')
    gen_layer0 = tl.layers.BatchNormLayer(gen_layer0, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='gen/layer0/batch_norm')
    gen_layer0.outputs = tf.nn.relu(gen_layer0.outputs, name='gen/layer0/relu')

    gen_layer1 = tl.layers.DeConv3dLayer(layer=gen_layer0, shape = [4,4,4,256,512], output_shape = [batch_size,8,8,8,256], strides=[1,2,2,2,1],
                                W_init = initializer, act=tf.identity, name='gen/layer1/decon2d')
    gen_layer1 = tl.layers.BatchNormLayer(gen_layer1, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='gen/layer1/batch_norm')
    gen_layer1.outputs = tf.nn.relu(gen_layer1.outputs, name='gen/layer1/relu')

    gen_layer2 = tl.layers.DeConv3dLayer(layer=gen_layer1, shape = [4,4,4,128,256], output_shape = [batch_size,16,16,16,128], strides=[1,2,2,2,1],
                                W_init = initializer, act=tf.identity, name='gen/layer2/decon2d')
    gen_layer2 = tl.layers.BatchNormLayer(gen_layer2, is_train=is_train, gamma_init=tf.random_normal_initializer(1., 0.02), name='gen/layer2/batch_norm')
    gen_layer2.outputs = tf.nn.relu(gen_layer2.outputs, name='gen/layer2/relu')

    gen_layer3 = tl.layers.DeConv3dLayer(layer=gen_layer2, shape = [4,4,4,1,128], output_shape = [batch_size,32,32,32,1], strides=[1,2,2,2,1],
                                W_init = initializer, act=tf.identity, name='gen/layer3/decon2d')
    gen_layer3.outputs = tf.nn.sigmoid(gen_layer3.outputs)


    return gen_layer3


def discriminator(inputs, batch_size=batch_size, is_train=True, reuse=False):

    if reuse:
        tf.get_variable_scope().reuse_variables()
        tl.layers.set_name_reuse(reuse)

    dis_in = tl.layers.InputLayer(inputs, name='dis/in')
    dis_layer0 = tl.layers.Conv3dLayer(dis_in, shape=[4,4,4,1,32], W_init = initializer, strides=[1,2,2,2,1], name='dis/layer0/conv2d')
    dis_layer0.outputs = tl.activation.leaky_relu(dis_layer0.outputs, alpha=0.2, name='dis/layer0/lrelu')

    dis_layer1 = tl.layers.Conv3dLayer(dis_layer0, shape=[4,4,4,32,64], W_init = initializer, strides=[1,2,2,2,1], name='dis/layer1/conv2d')
    dis_layer1 = tl.layers.BatchNormLayer(dis_layer1, is_train=is_train, name='dis/layer1/batch_norm')
    dis_layer1.outputs = tl.activation.leaky_relu(dis_layer1.outputs, alpha=0.2, name='dis/layer1/lrelu')

    dis_layer2 = tl.layers.Conv3dLayer(dis_layer1, shape=[4,4,4,64,128],  W_init = initializer, strides=[1,2,2,2,1], name='dis/layer2/conv2d')
    dis_layer2 = tl.layers.BatchNormLayer(dis_layer2, is_train=is_train, name='dis/layer2/batch_norm')
    dis_layer2.outputs = tl.activation.leaky_relu(dis_layer2.outputs, alpha=0.2, name='dis/layer2/lrelu')

    dis_layer3 = tl.layers.Conv3dLayer(dis_layer2, shape=[4,4,4,128,256],W_init = initializer, strides=[1,2,2,2,1], name='dis/layer3/conv2d')
    dis_layer3 = tl.layers.BatchNormLayer(dis_layer3, is_train=is_train, name='dis/layer3/batch_norm')
    dis_layer3.outputs = tl.activation.leaky_relu(dis_layer3.outputs, alpha=0.2, name='dis/layer3/lrelu')

    dis_flat = tl.layers.FlattenLayer(dis_layer3, name='dis/layer4/flatten')
    dis_output = tl.layers.DenseLayer(dis_flat, n_units=1, act=tf.identity, W_init=initializer, name='dis/layer4/lin_sigmoid')

    logits = dis_output.outputs
    dis_output.outputs = tf.nn.sigmoid(dis_output.outputs)

    return dis_output, logits

def train(checkpoint = None):
    z_size = 100
    z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32, name='z_vectors')
    x_vector = tf.placeholder(shape=[batch_size,32,32,32,1],dtype=tf.float32, name='real_chairs')

    _generator = generator(z_vector, is_train=True, reuse=False)
    #print 'generator: ',_generator
    _discriminator_x, dis_output_x = discriminator(x_vector, is_train=True, reuse=False)
    #print 'discriminator: ',_discriminator_x
    dis_output_x = tf.maximum(tf.minimum(dis_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("discriminator x probabilities", dis_output_x)

    _discriminator_z, dis_output_z = discriminator(_generator.outputs, is_train=True, reuse=True)
    dis_output_z = tf.maximum(tf.minimum(dis_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("discriminator z probabilities", dis_output_z)

    #d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_output_x, labels = tf.ones_like(_discriminator_x.outputs)) +\
                #tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_output_z, labels = tf.zeros_like(_discriminator_z.outputs)))
    #g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = dis_output_z, labels = tf.ones_like(_discriminator_z.outputs)))

    d_loss = -tf.reduce_mean(tf.log(dis_output_x) + tf.log(1-dis_output_z)) # loss for discriminator
    summary_d_loss = tf.summary.scalar("d_loss", d_loss)

    g_loss = -tf.reduce_mean(tf.log(dis_output_z)) # loss for generator
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)

    net_g_test = generator(z_vector, is_train=False, reuse=True)

    para_g=list(np.array(tf.trainable_variables())[[0,1,4,5,8,9,12,13]])
    para_d=list(np.array(tf.trainable_variables())[[14,15,16,17,20,21,24,25,28,29]])

    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=alpha_d,beta1=beta1).minimize(d_loss,var_list=para_d)
        # only update the weights for the generator network
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=alpha_g,beta1=beta1).minimize(g_loss,var_list=para_g)


        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        saver = tf.train.Saver(max_to_keep=25)

        with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
            # set GPU memeory fraction
            #tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.99)

            # variables need to be initialized before we can use them
            sess.run(tf.initialize_all_variables())
            if checkpoint is not None:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            start = time.time()
            z_sample = np.random.normal(0, 0.33, size=[batch_size,z_size]).astype(np.float32) # gaussian distribution between [-1, 1]

            for epoch in range(12001, n_epochs):
                x = get_batch(batch_size)
                z = np.random.normal(0, 0.33, size=[batch_size,z_size]).astype(np.float32)

                d_summary_merge = tf.summary.merge([summary_d_loss, summary_d_x_hist,summary_d_z_hist])
                summary_d,discriminator_loss = sess.run([d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                summary_g,generator_loss = sess.run([summary_g_loss,g_loss],feed_dict={z_vector:z})
                if not os.path.exists("./loss_write/"):
                    os.makedirs("./loss_write/")
                outfile = open("./loss_write/dg_loss%d.log"%epoch, "w")
                outfile.write(str(discriminator_loss))
                outfile.write(" ")
                outfile.write(str(generator_loss))
                outfile.close()
                if discriminator_loss <= 4.6*0.1:
                    sess.run([optimizer_op_g],feed_dict={z_vector:z})
                    print "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss

                elif generator_loss <= 4.6*0.1:
                    sess.run([optimizer_op_d],feed_dict={z_vector:z, x_vector:x})
                    print "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss
                else:
                    sess.run([optimizer_op_d],feed_dict={z_vector:z, x_vector:x})
                    sess.run([optimizer_op_g],feed_dict={z_vector:z})
                if epoch % 5 == 0:
                    summary_writer.add_summary(summary_d, epoch)
                    summary_writer.add_summary(summary_g, epoch)

                if epoch % 10 == 0:
                    time_lapse = time.time()-start
                    start = time.time()
                    print "epoch: ", epoch,", time spent: %.2fs" % time_lapse
                    print "d_loss: ", discriminator_loss, " g_loss: ", generator_loss

                if epoch % 500 == 0:
                    g_chairs = sess.run(net_g_test.outputs,feed_dict={z_vector:z_sample})
                    if not os.path.exists(train_sample_directory):
                        os.makedirs(train_sample_directory)
    
                    g_chairs.dump(train_sample_directory+'/'+str(epoch))
                if epoch % 100 == 0:
                    if not os.path.exists(model_directory):
                        os.makedirs(model_directory)

                    # save the trained model at different epoch
                    saver.save(sess, save_path = model_directory + '/' + str(epoch) + '.ckpt')

            print "Done"

if __name__ == "__main__":
    train('./models/')
    #train()
