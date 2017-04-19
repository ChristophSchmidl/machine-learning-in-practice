import os

from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Activation
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
from utils import *
from vgg16bn import Vgg16BN

ROOT_DIR = os.getcwd()
DATA_HOME_DIR = ROOT_DIR + '/data'

# paths
data_path = DATA_HOME_DIR + '/'
split_train_path = data_path + '/train/'
full_train_path = data_path + '/train_full/'
valid_path = data_path + '/valid/'
test_path = DATA_HOME_DIR + '/test/'
model_path = ROOT_DIR + '/models/vggbn_conv_640x360/'
submission_path = ROOT_DIR + '/submissions/vggbn_conv_640x360/'

# data
batch_size = 32
nb_split_train_samples = 3327
nb_full_train_samples = 3777
nb_valid_samples = 450
nb_test_samples = 1000
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
nb_classes = len(classes)

# model
nb_aug = 5
n_filters = 158
lr = 1e-3
dropout = 0.5
clip = 0.01

def get_classes(trn_path, val_path, test_path):
    batches = get_batches(trn_path, shuffle=False, batch_size=1)
    val_batches = get_batches(val_path, shuffle=False, batch_size=1)
    test_batches = get_batches(test_path, shuffle=False, batch_size=1)
    return (val_batches.classes, batches.classes, onehot(val_batches.classes), onehot(batches.classes), val_batches.filenames, batches.filenames, test_batches.filenames)

(val_classes, trn_classes, val_labels, trn_labels,
    val_filenames, filenames, test_filenames) = get_classes(split_train_path, valid_path, test_path)

vgg640 = Vgg16BN((360, 640)).model
vgg640.pop()
vgg640.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

'''
batches = get_batches(split_train_path, batch_size=1, target_size=(360, 640), shuffle=False, class_mode=None)
conv_trn_feat = vgg640.predict_generator(batches, nb_split_train_samples)
save_array(data_path + 'precomputed/trn_ft_640.dat', conv_trn_feat)

del conv_trn_feat


val_batches = get_batches(valid_path, batch_size=1, target_size=(360, 640), shuffle=False, class_mode=None)
conv_val_feat = vgg640.predict_generator(val_batches, nb_valid_samples)
save_array(data_path + 'precomputed/val_ft_640.dat', conv_val_feat)

del conv_val_feat

test_batches = get_batches(test_path, batch_size=1, target_size=(360, 640), shuffle=False, class_mode=None)
conv_test_feat = vgg640.predict_generator(test_batches, nb_test_samples)
save_array(data_path+'precomputed/test_ft_640.dat', conv_test_feat)

del conv_test_feat
'''

conv_val_feat = load_array(data_path + 'precomputed/val_ft_640.dat')
conv_trn_feat = load_array(data_path + 'precomputed/trn_ft_640.dat')
conv_test_feat = load_array(data_path + 'precomputed/test_ft_640.dat')

conv_layers, _ = split_at(vgg640, Convolution2D)

def get_lrg_layers():
    return [
        BatchNormalization(axis=1, input_shape=conv_layers[-1].output_shape[1:]),
        Convolution2D(n_filters, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(n_filters, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D(),
        Convolution2D(n_filters, 3, 3, activation='relu', border_mode='same'),
        BatchNormalization(axis=1),
        MaxPooling2D((1, 2)),
        Convolution2D(8, 3, 3, border_mode='same'),
        Dropout(dropout),
        GlobalAveragePooling2D(),
        Activation('softmax')
    ]

lrg_model = Sequential(get_lrg_layers())

#lrg_model.summary()

lrg_model.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

model_fn = model_path + '{val_loss:.2f}-loss_{epoch}epoch_640x360_vgg16bn.h5'
ckpt = ModelCheckpoint(filepath=model_fn, monitor='val_loss', save_best_only=True, save_weights_only=True)


lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=2, verbose=2, validation_data=(conv_val_feat, val_labels), callbacks=[ckpt])

lrg_model.optimizer.lr /= 10

lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=5, verbose=2, validation_data=(conv_val_feat, val_labels), callbacks=[ckpt])


lrg_model.optimizer.lr /= 10

lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=5, verbose=2, validation_data=(conv_val_feat, val_labels), callbacks=[ckpt])


lrg_model.optimizer.lr /= 10

lrg_model.fit(conv_trn_feat, trn_labels, batch_size=batch_size, nb_epoch=5, verbose=2, validation_data=(conv_val_feat, val_labels), callbacks=[ckpt])


def gen_preds_from_saved(use_all=True, weights_file=None):
    model = Sequential(get_lrg_layers())

    if use_all:
        preds = np.zeros((nb_test_samples, nb_classes))

        for root, dirs, files in os.walk(model_path):
            n_mods = 0
            for f in files:
                model.load_weights(model_path + f)
                preds += model.predict(conv_test_feat, batch_size=batch_size)
                n_mods += 1

        preds /= n_mods

    else:
        model.load_weights(model_path + weights_file)
        preds = model.predict(conv_test_feat, batch_size=batch_size)

    return preds


def gen_preds(model):
    if nb_aug:

        gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.05, zoom_range=0.05,
                                 channel_shift_range=10, height_shift_range=0.05, shear_range=0.05,
                                 horizontal_flip=True)
        predictions = np.zeros(shape=(nb_test_samples, nb_classes))

        for aug in range(nb_aug):
            test_batches = get_batches(test_path, batch_size=1, target_size=(360, 640), shuffle=False,
                                       class_mode=None, gen=gen)
            conv_test_feat = vgg640.predict_generator(test_batches, nb_test_samples)
            predictions += model.predict(conv_test_feat, batch_size=batch_size)

        predictions /= nb_aug

    else:
        predictions = model.predict(conv_test_feat, batch_size=batch_size)

    return predictions


# preds = gen_preds_from_saved(use_all=True, weights_file=None)
preds = gen_preds(lrg_model)



def write_submission(predictions, filenames):
    preds = np.clip(predictions, clip, 1-clip)
    sub_fn = submission_path + '{0}-aug_{1}clip_vgg_bn'.format(nb_aug, clip)

    with open(sub_fn + '.csv', 'w') as f:
        print("Writing Predictions to CSV...")
        f.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
        for i, image_name in enumerate(filenames):
            pred = ['%.6f' % p for p in preds[i, :]]
            f.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))
        print("Done.")

write_submission(preds, test_filenames)