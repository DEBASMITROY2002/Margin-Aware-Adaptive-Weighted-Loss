from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras as K
import tensorflow as tf
import tensorflow.keras as tk

def data_generator(runtimename, BATCH_SIZE, HEIGHT, WIDTH):
    train_datagen = ImageDataGenerator(
        rescale = 1./ 255.
    )
 
    valid_datagen = ImageDataGenerator(
        rescale = 1./ 255.
    )

    if runtimename == 'aptos':
        pass
    if runtimename == 'cifar10':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar10/cifar10_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar10/cifar10_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator;

    if runtimename == 'cifar50':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar10/cifar50_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar10/cifar50_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator;
    if runtimename == 'cifar100':
        train_generator = train_datagen.flow_from_directory(
            'Data/cifar10/cifar100_train',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );

        validation_generator = valid_datagen.flow_from_directory(
            'Data/cifar10/cifar100_test',
            batch_size = BATCH_SIZE,
            class_mode = 'categorical',
            target_size = (HEIGHT,WIDTH)
        );
        return train_generator,validation_generator;
    if runtimename == 'fmnist10':
        pass
    if runtimename == 'fmnist50':
        pass
    if runtimename == 'fmnist100':
        pass
    if runtimename == 'ham10000':
        pass
    