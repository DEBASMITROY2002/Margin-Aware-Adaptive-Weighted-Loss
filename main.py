import argparse
import tensorflow as tf
import tensorflow.keras as tk
import numpy as np
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
import zipfile
from utils.model import *
from utils.metric import *
from utils.data_generator import *
from MAAW import *

parser = argparse.ArgumentParser()
parser.add_argument('--runtimename', type=str, default = './', help='Directory where the image data is stored')
# parser.add_argument('--epochs', type=int, default = 10, help='Number of Epochs of training')
# parser.add_argument('--batch_size', type=int, default = 32, help='Batch size for training')
# parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning Rate')
# parser.add_argument('--stepLR', type=int, default=5, help='Step size for Step LR scheduler')
args = parser.parse_args()

runtimename = args.runtimename;
model_save_dir = '/saved_models/'

"""

##Data Batching"""

BATCH_SIZE = 256
HEIGHT = 32
WIDTH = 32
INIT_LEARNING_RATE = 1e-4

train_generator,validation_generator = data_generator(BATCH_SIZE, HEIGHT, WIDTH)

auto_t_steps_per_epoch = train_generator.n//BATCH_SIZE
auto_v_steps_per_epoch = validation_generator.n//BATCH_SIZE
NUM_CLASS = train_generator.num_classes
test_generator = validation_generator

train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
val_acc_metric   = tf.keras.metrics.CategoricalAccuracy()

"""## Run"""

model = getmodel(runtimename, NUM_CLASS)

x_test, y_test = (next(test_generator))

for i in range(auto_v_steps_per_epoch):
 x_test_temp, y_test_temp = (next(test_generator))
 x_test = np.concatenate((x_test,x_test_temp))
 y_test = np.concatenate((y_test,y_test_temp))

MODELS = []
HISTORYS = []
STAGE = 0

while True:
  print("========== STAGE "+str(STAGE)+" ==========")
  STAGE = STAGE+1
  alpha = float(input("alpha: "))
  beta = float(input("beta: "))
  max_epoch = int(input("max_epoch: "))
  val_acc_thresh = float(input("val_acc_thresh: "))
  loss_object = SparseCategorical_LSM_DWB_Loss(maj_wt=alpha,min_wt=beta)
  history = train(model,loss_object, train_generator , validation_generator, train_acc_metric, val_acc_metric, epochs = max_epoch,t_steps_per_epoch=5,v_steps_per_epoch=5,val_acc_threshold=val_acc_thresh,INIT_LEARNING_RATE=INIT_LEARNING_RATE)
  MODELS.append(model)
  HISTORYS.append(history)
  model.save(model_save_dir+runtimename+str(alpha)+"_"+str(beta)+".h5")
  print("[ Model Saved As "+model_save_dir+runtimename+str(alpha)+"_"+str(beta)+".h5 ]")
  computePerformance(x_test, y_test, model)
  if input('Do You Want To Continue? y/n') != 'y':
    break