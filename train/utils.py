
import math
#from net import createModel, defineExperimentPaths
from keras.callbacks import (EarlyStopping, LearningRateScheduler)
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.metrics import confusion_matrix, average_precision_score
import random

def label_one_hot(label_list):
    label = []
    for i in label_list:
        if i=='0':
            label.append([1,0])
        else:
            label.append([0,1])
    return label


def xl_kmer(window_size, npz_file_path):
    xl_72 = np.load(npz_file_path, allow_pickle=True)
    seq_len_list = []
    vec = []
    for i in xl_72.keys():
        vec.append(xl_72[i])
        seq_len_list.append(xl_72[i].shape[0])
    count = 0
    win_vec = []
    for i in seq_len_list:
        for j in range(i):
            win_start = j - window_size
            win_end = j + window_size+1
            if win_start < 0 :
                current_vec = np.concatenate([np.zeros((-win_start,1024),dtype=float),vec[count][0:win_end]],axis=0)
                #print(current_vec.shape)
            elif win_end > i:
                current_vec = np.concatenate([vec[count][win_start:i], np.zeros((win_end-i, 1024), dtype=float)])
                #print(current_vec.shape)
            else:
                current_vec = vec[count][win_start:win_end]
            win_vec.append(current_vec.reshape(window_size*2+1,1024))
        count+=1
    return np.array(win_vec)
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale
def softmax(x, axis=1):
    # 计算每行的最大值
    row_max1 = x.max(axis=axis)
 
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max1 = row_max1.reshape(-1, 1)
    x = x - row_max1
 
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s
def evaluate(y_true_prob,y_pred_prob):
  if y_pred_prob[0][1]+y_pred_prob[0][0] != 1.0:
    y_pred_prob = softmax(np.array(y_pred_prob))
  y_pred = np.argmax(y_pred_prob,axis=1)
  y_true = np.argmax(y_true_prob,axis=1)
  acc = accuracy_score(y_true,y_pred)
  mcc = matthews_corrcoef(y_true,y_pred)
  pre = precision_score(y_true,y_pred)
  recall = recall_score(y_true,y_pred)
  f1 = f1_score(y_true,y_pred)
  auroc = roc_auc_score(y_true, y_pred_prob[:,1])
  a = confusion_matrix(y_true, y_pred)
  TP = a[1][1]
  FP = a[0][1]
  TN = a[0][0]
  FN = a[1][0]
  sens = TP/(TP+FN)
  Spec = TN/(TN+FP)
  auprc = average_precision_score(y_true, y_pred_prob[:,1])
  print("%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f"%(len(y_pred),sens,Spec,pre,acc,f1,mcc,auroc,auprc))
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.6
    epochs_drop = 6.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    print(lrate)
    return lrate
def callbacks():
  callbacks = [
      EarlyStopping(monitor='val_loss', patience=6),
                      #verbose=2, mode='auto'),
      # ModelCheckpoint(checkpoint_weight,
      #                 monitor='val_loss',
      #                 verbose=1,
      #                 save_best_only=True,
      #                 mode='auto',
      #                 period=1),
      LearningRateScheduler(step_decay)
  ]
  return callbacks

def data_split(vec_xl_1, vec_bio_1, emsemble_num):
  label1 = np.load('./xl_model/label_843.npy')
  label2 = np.load('./xl_model/label_186.npy')
  label = np.concatenate([label1,label2],axis=0)
  negative_list = []
  positive_list = []
  for i in range(len(label)):
    if label[i] == '0':
      negative_list.append(i)
    else:
      positive_list.append(i)
  random.shuffle(negative_list)
  split_num = emsemble_num#round(list(label).count('0')/list(label).count('1'))-3
  sample_num = list(label).count('0')//split_num
  print(len(positive_list))
  sub_list_xl = []
  sub_list_bio = []
  positive_list_xl = vec_xl_1[positive_list]
  positive_list_bio = vec_bio_1[positive_list]
  for i in range(split_num):
    start = i*sample_num
    end = (i+1)*sample_num
    if i == split_num-1:
      end = len(negative_list)
    sub_list_xl.append(vec_xl_1[negative_list[start:end]])
    sub_list_bio.append(vec_bio_1[negative_list[start:end]])
  return positive_list_xl, positive_list_bio, sub_list_xl, sub_list_bio
def label_sum(pre,now):
  c = []
  for i in range(len(now)):
    c.append(np.sum((pre[i],now[i]),axis=0))
  return c
