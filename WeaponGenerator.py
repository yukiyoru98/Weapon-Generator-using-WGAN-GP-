import os
import os.path as path
import numpy as np
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Activation, ZeroPadding2D
from keras.layers.merge import _Merge
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image

#parameters
training_ratio = 5
GRADIENT_PENALTY_WEIGHT = 10 
latent_dim = 100
img_rows = 128
img_cols = 128
img_channel = 4
epochs = 10
batch_size = 50
save_model_interval = 2
save_img_interval = 2
d_filename = 'd_model'
g_filename = 'g_model'
output_dir = 'OutputImages/' # path to generator's output images
real_images_dir = 'RealImages/' # path to real images
model_dir = 'TrainedModel/' # path to save model

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples,
                          gradient_penalty_weight):
    
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)


def make_generator():
    model = Sequential()

    model.add(Dense(64 * 16 *16, activation="relu", input_dim=latent_dim))
    model.add(Reshape((16, 16, 64)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(img_channel, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)


def make_discriminator():
    img_shape = (img_rows, img_cols, img_channel)
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((batch_size, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def plot_save_images(images, epoch_index, name): #plot the first 10 images into a 2x5 figure and save as png
    img_filename = output_dir + name + str(epoch_index) + '.png'
    images = (images + 1) * 0.5
   
    plt.figure(figsize=(12,7))
    
    for i in range(0, 10):
        
        plt.subplot(2, 5, 1+i)
        plt.imshow(images[i, :, :, :])
        plt.xticks([])
        plt.yticks([])
        i += 1
       
    if os.path.isfile(img_filename):
        os.remove(img_filename) 
    plt.savefig(img_filename)
       
def load_imgs(imdir): # load real images for training
    result = []
    for img_file in os.listdir(imdir):
        img = Image.open(path.join(imdir, img_file))
        img = np.array(img)
        result.append(img)
    return np.stack(result)

def train_model(): #compile and train 
    # initialize the generator and discriminator.
    generator = make_generator()
    discriminator = make_discriminator()

    #build discriminator_model
    generator.trainable = False
    real_input = Input(shape = (img_rows, img_cols, img_channel))
    fake_noise_input = Input(shape = (latent_dim, ))
    fake_input = generator(fake_noise_input)
    real_output = discriminator(real_input)
    fake_output = discriminator(fake_input)
    #output (avg)
    averaged_samples = RandomWeightedAverage()([real_input, fake_input])
    avg_output = discriminator(averaged_samples)
    # gradient penalty loss function 
    partial_gp_loss = partial(gradient_penalty_loss,
                            averaged_samples=averaged_samples,
                            gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
                            
    partial_gp_loss.__name__ = 'gradient_penalty' # Functions need names or Keras will throw an error
    discriminator_model = Model([real_input, fake_noise_input], [real_output, fake_output, avg_output])
    
    #build g_model
    discriminator.trainable = False
    generator.trainable = True
    gen_noise_input = Input(shape=(latent_dim,))
    gen_fake_img = generator(gen_noise_input)
    gen_output = discriminator(gen_fake_img)
    generator_model = Model(gen_noise_input, gen_output)
    
    #compile models
    discriminator_model.compile(optimizer = RMSprop(lr=0.00005),
                                loss=[wasserstein_loss,
                                    wasserstein_loss,
                                    partial_gp_loss],
                                metrics=['accuracy'])
    generator_model.compile(optimizer = RMSprop(lr=0.00005), loss=wasserstein_loss, metrics=['accuracy'])

    #load real images
    real_images = load_imgs(real_images_dir)
    real_images = real_images.astype('float32')
    real_images = real_images / 127.5 - 1
    
    # labels for the outputs
    real_labels = -np.ones((batch_size, 1))
    fake_labels = -real_labels 
    dummy_labels = np.zeros((batch_size, 1)) #for avg

    batch_nums = (int)(real_images.shape[0] / batch_size)
    print("Number of batches per epoch: ", batch_nums) 

    for epoch in range(epochs):
        
        np.random.shuffle(real_images)
        
        for i in range(batch_nums):
        
            #=======train D========
            for _ in range(training_ratio):

                idx = np.random.randint(0, real_images.shape[0], batch_size)
                real_image_batch = real_images[idx]
                noise = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))# generate fake images
                d_loss = discriminator_model.train_on_batch([real_image_batch, noise], [real_labels, fake_labels, dummy_labels])

            #=======train G========
            noise_train = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
            g_loss = generator_model.train_on_batch(noise_train, real_labels)
        
        #======================
        print("Epoch: " + str(epoch) )
        print(f'[D_loss_real: {d_loss[0]:.5f} | D_loss_fake: {d_loss[1]:.3f}]')
        print(f'[G_loss: {g_loss[0]:.5f} | G_acc: {g_loss[1]:.3f}]')

        if epoch % save_img_interval == 0: # result of generator for current epoch
            noise_epoch = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))
            gen_img = generator.predict(noise_epoch)
            plot_save_images(gen_img, epoch, 'fake')

        if epoch % save_model_interval == 0: # save models and weights
            discriminator_model.save(model_dir + str(epoch) + '_' + d_filename + '.h5')
            generator_model.save(model_dir + str(epoch) + '_' + g_filename + '.h5')

            discriminator_model.save_weights(model_dir + str(epoch) + '_' + d_filename + '_weights.h5')
            generator_model.save_weights(model_dir + str(epoch) + '_' + g_filename + '_weights.h5')

if __name__ == '__main__':
    train_model()
    