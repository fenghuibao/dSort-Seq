import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import softmax
import os
import argparse
import pickle

tf.keras.backend.set_floatx('float64')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

BUFFER_SIZE = 1000000
BATCH_SIZE = 6400

parser = argparse.ArgumentParser('Parameter estimation of FACS-seq')
parser.add_argument('-p' ,'--pi', help='Propotion of each sample in original library', required=True)
parser.add_argument('-f', '--frac', help='Binned distribution', required=True)
parser.add_argument('-d', '--data', help='cytometry data of the library', required=True)
parser.add_argument('-b', '--boundary', help='FACS boundaries', required=True)
args = parser.parse_args()

pi = pd.read_csv(args.pi, header=None)
frac = pd.read_csv(args.frac, index_col=0)
data = pd.read_csv(args.data, header=None)
gate = pd.read_csv(args.boundary, header=None)

pi = tf.cast(pi[1].values, tf.float64)
frac = tf.cast(frac.values, tf.float64)
data = tf.cast(data[0].values, tf.float64)
data = tf.reshape(data,[-1,1])
gate = tf.cast(gate[1].values, tf.float64)


FACS_data = tf.data.Dataset.from_tensor_slices(data).repeat(2).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
n_comp = 2

class FS_generator(tf.keras.Model):
    def __init__(self, pi, gate, batch_size):
        super().__init__()
        self.pi = pi
        self.gate = gate
        self.batch_size = batch_size
        self.lamb = tf.Variable(tf.random.normal([pi.shape[0], n_comp], mean=0.5, stddev=0.1, dtype=tf.float64), trainable=True)
        self.mu = tf.Variable(tf.random.normal([pi.shape[0], n_comp], mean=0.3, stddev=0.4, dtype=tf.float64), trainable=True)
        self.sigma = tf.Variable(tf.random.normal([pi.shape[0], n_comp], mean=0.2, stddev=0.1, dtype=tf.float64), trainable=True, constraint=lambda x: tf.clip_by_value(x, 0.05, 2))
    def __call__(self):
        # Generate samples
        dist = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(probs=self.pi),
                                                                                             components_distribution=tfp.distributions.MixtureSameFamily(
                                                                                             mixture_distribution=tfp.distributions.Categorical(logits=self.lamb),
                                                                                             components_distribution=tfp.distributions.Normal(loc=self.mu, scale=self.sigma)))
        sample = dist.sample(sample_shape=[self.batch_size,1])

        # Calculate cumulative distribution function
        dist_ = tfp.distributions.MixtureSameFamily(mixture_distribution=tfp.distributions.Categorical(logits=self.lamb),
                                                                                                components_distribution=tfp.distributions.Normal(loc=self.mu, scale=self.sigma))
        dist_cdf = tf.concat([[dist_.cdf(g)] for g in self.gate.numpy()],axis=0)
        cdf_diff = tf.concat([tf.reshape(dist_cdf[0],[1,-1]), dist_cdf[1:]-dist_cdf[:-1]], axis=0)
        cdf_diff_tail = tf.reshape(1 - tf.reduce_sum(cdf_diff, axis=0), [1,-1])
        cdf_diff = tf.transpose(tf.concat([cdf_diff,cdf_diff_tail], axis=0))
        return sample, cdf_diff
generator = FS_generator(pi, gate, BATCH_SIZE)

def FS_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1))
    return model
discriminator = FS_discriminator()

# loss function
BinaryCrossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss =  BinaryCrossentropy(tf.ones_like(real_output), real_output)
    fake_loss = BinaryCrossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output, cdf_diff):
    dist_loss = tf.math.reduce_sum(pi * tf.keras.losses.categorical_crossentropy(frac, cdf_diff))
    gen_loss = BinaryCrossentropy(tf.ones_like(fake_output), fake_output)
    return dist_loss, gen_loss

# optimizer
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

def distribution_plot(lamb, mu, sigma, image_name):
    x = np.linspace(-2.7,1,500)
    fig, ax = plt.subplots()
    ax.hist(np.squeeze(data), bins=x, density=True,color='r', alpha=0.5, label='real_distribution')
    prob = softmax(lamb, axis=1)
    y = [pi.numpy().dot((prob * norm.pdf(i, loc=mu, scale=sigma)).sum(axis=1)) for i in x]
    ax.plot(x, y, color='b', alpha=0.5, label='generated_distribution')
    ax.set_title(image_name, fontsize='x-large')
    plt.legend(loc='best')
    plt.savefig('%s.png'%image_name)
    plt.close(fig)


def train_step(real_sample):
    for i in range(8):
        with tf.GradientTape() as disc_tape:
            fake_sample, dist_cdf = generator()
            fake_output = discriminator(fake_sample)
            real_output = discriminator(real_sample)
            disc_loss = discriminator_loss(real_output, fake_output)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    with tf.GradientTape() as gen_tape:
        fake_sample, dist_cdf = generator()
        fake_output = discriminator(fake_sample)
        dist_loss, gen_loss = generator_loss(fake_output, dist_cdf)
        total_loss = 0.995 * dist_loss + 0.005 * gen_loss
    gen_gradients = gen_tape.gradient(total_loss, [generator.lamb, generator.mu, generator.sigma])
    generator_optimizer.apply_gradients(zip(gen_gradients, [generator.lamb, generator.mu, generator.sigma]))
    return disc_loss, dist_loss, gen_loss

log_dir = 'logs'
summary_writer = tf.summary.create_file_writer(log_dir)

def train(FACS_data):
    num_iter = 0
    min_dist_loss = 10
    for real_sample in FACS_data:
        num_iter += 1
        disc_loss, dist_loss, gen_loss = train_step(real_sample)
        
        mu = generator.mu.numpy()
        sigma = generator.sigma.numpy()
        lamb = generator.lamb.numpy()

        if num_iter % 50 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('discriminator_loss', disc_loss, step=num_iter)
                tf.summary.scalar('distribution_loss', dist_loss, step=num_iter)
                tf.summary.scalar('generator_loss', gen_loss, step=num_iter)
        print('num_iter:%d, disc_loss:%f, dist_loss:%f, gen_loss:%f'%(num_iter, disc_loss, dist_loss, gen_loss))


        #distribution_plot(lamb, mu, sigma, image_name='num_iter%d'%num_iter)
        if dist_loss < min_dist_loss:
            min_dist_loss = dist_loss
            with open('mu.pickle', 'wb') as f:
                    pickle.dump(mu,f)
            with open('sigma.pickle', 'wb') as f:
                    pickle.dump(sigma,f)
            with open('lamb.pickle', 'wb') as f:
                    pickle.dump(lamb,f)

train(FACS_data)
