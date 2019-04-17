import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import time
e_start = 15400
n_epochs = 30001
batch_size = 64
alpha_g = 0.0015
alpha_d = 0.000024
alpha_e = 0.0001
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

def encoder(inputs, batch_size=batch_size, is_train=True, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
        tl.layers.set_name_reuse(reuse)

    enc_in = tl.layers.InputLayer(inputs, name='enc/in')
    enc_layer0 = tl.layers.Conv3dLayer(enc_in, shape=[4,4,4,1,32], W_init = initializer, strides=[1,2,2,2,1], name='enc/layer0/conv3d')
    enc_layer0.outputs = tl.activation.leaky_relu(enc_layer0.outputs, alpha=0.2, name='enc/layer0/lrelu')

    enc_layer1 = tl.layers.Conv3dLayer(enc_layer0, shape=[4,4,4,32,64], W_init = initializer, strides=[1,2,2,2,1], name='enc/layer1/conv3d')
    enc_layer1 = tl.layers.BatchNormLayer(enc_layer1, is_train=is_train, name='enc/layer1/batch_norm')
    enc_layer1.outputs = tl.activation.leaky_relu(enc_layer1.outputs, alpha=0.2, name='enc/layer1/lrelu')

    enc_layer2 = tl.layers.Conv3dLayer(enc_layer1, shape=[4,4,4,64,128],  W_init = initializer, strides=[1,2,2,2,1], name='enc/layer2/conv3d')
    enc_layer2 = tl.layers.BatchNormLayer(enc_layer2, is_train=is_train, name='enc/layer2/batch_norm')
    enc_layer2.outputs = tl.activation.leaky_relu(enc_layer2.outputs, alpha=0.2, name='denclayer2/lrelu')

    enc_layer3 = tl.layers.Conv3dLayer(enc_layer2, shape=[4,4,4,128,256],W_init = initializer, strides=[1,2,2,2,1], name='enc/layer3/conv3d')
    enc_layer3 = tl.layers.BatchNormLayer(enc_layer3, is_train=is_train, name='enc/layer3/batch_norm')
    enc_layer3.outputs = tl.activation.leaky_relu(enc_layer3.outputs, alpha=0.2, name='enc/layer3/lrelu')

    enc_flat = tl.layers.FlattenLayer(enc_layer3, name='enc/layer4/flatten')
    #enc_output = tl.layers.DenseLayer(enc_flat, n_units=1, act=tf.identity, W_init=initializer, name='enc/layer4/lin_sigmoid')
    enc_mn = tl.layers.DenseLayer(enc_flat, n_units=100, act=tf.identity, W_init=initializer, name='enc/layer4/lin_mn')
    enc_sd = tl.layers.DenseLayer(enc_flat, n_units=100, act=tf.identity, W_init=initializer, name='enc/layer4/lin_sd')
    epsilon = tf.random_normal(tf.stack([batch_size, 100]))
    z = enc_mn.outputs + tf.multiply(epsilon, tf.exp(enc_sd.outputs))
    #logits = enc_output.outputs
    #enc_output.outputs = tf.nn.sigmoid(enc_output.outputs)

    return z, enc_mn.outputs, enc_sd.outputs


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
    #z_vector = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32, name='z_vectors')
    x_vector = tf.placeholder(shape=[batch_size,32,32,32,1],dtype=tf.float32, name='real_chairs')
    y = tf.placeholder(shape=[batch_size,32,32,32,1],dtype=tf.float32, name='Y')
    y_flat = tf.reshape(y, shape = [-1, 32 * 32 * 32])
    z_vector, mn, sd = encoder(x_vector)
    
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

    unreshaped = tf.reshape(_generator.outputs, [-1, 32 * 32 * 32])
    img_loss = tf.reduce_sum(tf.squared_difference(unreshaped, y_flat), 1)
    latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * sd - tf.square(mn) - tf.exp(2.0 * sd), 1)
    e_loss = tf.reduce_mean(img_loss + latent_loss)
    
    d_loss = -tf.reduce_mean(tf.log(dis_output_x) + tf.log(1-dis_output_z)) # loss for discriminator
    summary_d_loss = tf.summary.scalar("d_loss", d_loss)

    g_loss = -tf.reduce_mean(tf.log(dis_output_z)) # loss for generator
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)

    net_g_test = generator(z_vector, is_train=False, reuse=True)

    """
    0<tf.Variable 'enc/layer0/conv3d/W_conv3d:0' shape=(4, 4, 4, 1, 32) dtype=float32_ref>,
    1<tf.Variable 'enc/layer0/conv3d/b_conv3d:0' shape=(32,) dtype=float32_ref>,
    2<tf.Variable 'enc/layer1/conv3d/W_conv3d:0' shape=(4, 4, 4, 32, 64) dtype=float32_ref>,
    3<tf.Variable 'enc/layer1/conv3d/b_conv3d:0' shape=(64,) dtype=float32_ref>,
    4<tf.Variable 'enc/layer1/batch_norm/beta:0' shape=(64,) dtype=float32_ref>,
    5<tf.Variable 'enc/layer1/batch_norm/gamma:0' shape=(64,) dtype=float32_ref>,
    6<tf.Variable 'enc/layer2/conv3d/W_conv3d:0' shape=(4, 4, 4, 64, 128) dtype=float32_ref>,
    7<tf.Variable 'enc/layer2/conv3d/b_conv3d:0' shape=(128,) dtype=float32_ref>,
    8<tf.Variable 'enc/layer2/batch_norm/beta:0' shape=(128,) dtype=float32_ref>,
    9<tf.Variable 'enc/layer2/batch_norm/gamma:0' shape=(128,) dtype=float32_ref>,
    10<tf.Variable 'enc/layer3/conv3d/W_conv3d:0' shape=(4, 4, 4, 128, 256) dtype=float32_ref>,
    11<tf.Variable 'enc/layer3/conv3d/b_conv3d:0' shape=(256,) dtype=float32_ref>,
    12<tf.Variable 'enc/layer3/batch_norm/beta:0' shape=(256,) dtype=float32_ref>,
    13<tf.Variable 'enc/layer3/batch_norm/gamma:0' shape=(256,) dtype=float32_ref>,
    14<tf.Variable 'enc/layer4/lin_mn/W:0' shape=(2048, 100) dtype=float32_ref>,
    15<tf.Variable 'enc/layer4/lin_mn/b:0' shape=(100,) dtype=float32_ref>,
    16<tf.Variable 'enc/layer4/lin_sd/W:0' shape=(2048, 100) dtype=float32_ref>,
    17<tf.Variable 'enc/layer4/lin_sd/b:0' shape=(100,) dtype=float32_ref>,
    
    18<tf.Variable 'gen/layer0/lin/W:0' shape=(100, 32768) dtype=float32_ref>,
    19<tf.Variable 'gen/layer0/lin/b:0' shape=(32768,) dtype=float32_ref>,
    <tf.Variable 'gen/layer0/batch_norm/beta:0' shape=(512,) dtype=float32_ref>,
    <tf.Variable 'gen/layer0/batch_norm/gamma:0' shape=(512,) dtype=float32_ref>,
    <tf.Variable 'gen/layer1/decon2d/W_deconv3d:0' shape=(4, 4, 4, 256, 512) dtype=float32_ref>,
    <tf.Variable 'gen/layer1/decon2d/b_deconv3d:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'gen/layer1/batch_norm/beta:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'gen/layer1/batch_norm/gamma:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'gen/layer2/decon2d/W_deconv3d:0' shape=(4, 4, 4, 128, 256) dtype=float32_ref>,
    <tf.Variable 'gen/layer2/decon2d/b_deconv3d:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'gen/layer2/batch_norm/beta:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'gen/layer2/batch_norm/gamma:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'gen/layer3/decon2d/W_deconv3d:0' shape=(4, 4, 4, 1, 128) dtype=float32_ref>,
    <tf.Variable 'gen/layer3/decon2d/b_deconv3d:0' shape=(1,) dtype=float32_ref>,
    
    <tf.Variable 'dis/layer0/conv2d/W_conv3d:0' shape=(4, 4, 4, 1, 32) dtype=float32_ref>,
    <tf.Variable 'dis/layer0/conv2d/b_conv3d:0' shape=(32,) dtype=float32_ref>,
    <tf.Variable 'dis/layer1/conv2d/W_conv3d:0' shape=(4, 4, 4, 32, 64) dtype=float32_ref>,
    <tf.Variable 'dis/layer1/conv2d/b_conv3d:0' shape=(64,) dtype=float32_ref>,
    <tf.Variable 'dis/layer1/batch_norm/beta:0' shape=(64,) dtype=float32_ref>,
    <tf.Variable 'dis/layer1/batch_norm/gamma:0' shape=(64,) dtype=float32_ref>,
    <tf.Variable 'dis/layer2/conv2d/W_conv3d:0' shape=(4, 4, 4, 64, 128) dtype=float32_ref>,
    <tf.Variable 'dis/layer2/conv2d/b_conv3d:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'dis/layer2/batch_norm/beta:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'dis/layer2/batch_norm/gamma:0' shape=(128,) dtype=float32_ref>,
    <tf.Variable 'dis/layer3/conv2d/W_conv3d:0' shape=(4, 4, 4, 128, 256) dtype=float32_ref>,
    <tf.Variable 'dis/layer3/conv2d/b_conv3d:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'dis/layer3/batch_norm/beta:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'dis/layer3/batch_norm/gamma:0' shape=(256,) dtype=float32_ref>,
    <tf.Variable 'dis/layer4/lin_sigmoid/W:0' shape=(2048, 1) dtype=float32_ref>,
    <tf.Variable 'dis/layer4/lin_sigmoid/b:0' shape=(1,) dtype=float32_ref>
    """
    #print tf.trainable_variables()
    para_e=list(np.array(tf.trainable_variables())[[0,1,2,3,6,7,10,11,14,15,16,17]])
    para_g=list(np.array(tf.trainable_variables())[[18,19,22,23,26,27,30,31]])
    para_d=list(np.array(tf.trainable_variables())[[32,33,34,35,38,39,42,43,46,47]])

    with tf.variable_scope(tf.get_variable_scope(), reuse = tf.AUTO_REUSE):
        optimizer_op_e = tf.train.AdamOptimizer(learning_rate = alpha_e, beta1 = beta1).minimize(e_loss, var_list = para_e)
        optimizer_op_d = tf.train.AdamOptimizer(learning_rate=alpha_d,beta1=beta1).minimize(d_loss,var_list=para_d)
        # only update the weights for the generator network
        optimizer_op_g = tf.train.AdamOptimizer(learning_rate=alpha_g,beta1=beta1).minimize(g_loss,var_list=para_g)


        #summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        saver = tf.train.Saver(max_to_keep=25)

        with tf.Session(config = tf.ConfigProto(gpu_options = tf.GPUOptions(allow_growth = True))) as sess:
            # set GPU memeory fraction
            #tl.ops.set_gpu_fraction(sess=sess, gpu_fraction=0.99)

            # variables need to be initialized before we can use them
            sess.run(tf.initialize_all_variables())
            if checkpoint is not None:
                saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
            start = time.time()
            #z_sample = np.random.normal(0, 0.33, size=[batch_size,z_size]).astype(np.float32) # gaussian distribution between [-1, 1]

            for epoch in range(e_start, n_epochs):
                x = get_batch(batch_size)
                #z = np.random.normal(0, 0.33, size=[batch_size,z_size]).astype(np.float32)

                d_summary_merge = tf.summary.merge([summary_d_loss, summary_d_x_hist,summary_d_z_hist])
                #summary_d,discriminator_loss = sess.run([d_summary_merge,d_loss],feed_dict={z_vector:z, x_vector:x})
                #summary_g,generator_loss = sess.run([summary_g_loss,g_loss],feed_dict={z_vector:z})
                encoder_loss, generator_loss, discriminator_loss = sess.run([e_loss, g_loss, d_loss], feed_dict = {x_vector:x, y:x})
                if not os.path.exists("./loss_write/"):
                    os.makedirs("./loss_write/")
                outfile = open("./loss_write/dge_loss%d.log"%epoch, "w")
                outfile.write(str(discriminator_loss))
                outfile.write(" ")
                outfile.write(str(generator_loss))
                outfile.write(" ")
                outfile.write(str(encoder_loss))
                outfile.close()
                sess.run([optimizer_op_e], feed_dict = {x_vector:x, y:x})
                if discriminator_loss <= 4.6*0.1:
                    sess.run([optimizer_op_g],feed_dict={x_vector:x, y:x})
                    print "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss

                elif generator_loss <= 4.6*0.1:
                    sess.run([optimizer_op_d],feed_dict={x_vector:x, y:x})
                    print "epoch: ",epoch,', d_loss:',discriminator_loss,'g_loss:',generator_loss
                else:
                    sess.run([optimizer_op_d],feed_dict={x_vector:x, y:x})
                    sess.run([optimizer_op_g],feed_dict={x_vector:x, y:x})
                """
                if epoch % 5 == 0:
                    summary_writer.add_summary(summary_d, epoch)
                    summary_writer.add_summary(summary_g, epoch)
                """
                
                if epoch % 10 == 0:
                    time_lapse = time.time()-start
                    start = time.time()
                    print "epoch: ", epoch,", time spent: %.2fs" % time_lapse
                    print "d_loss: ", discriminator_loss, " g_loss: ", generator_loss, "e_loss: ", encoder_loss

                if epoch % 500 == 0:
                    g_chairs = sess.run(net_g_test.outputs,feed_dict={x_vector:x, y:x})
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
