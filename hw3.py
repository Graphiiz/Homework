import torch
import scipy.io
import numpy as np
from collections import *
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import json
import os.path

def get_variance_and_kernel(X):
    sum = 0
    K = np.zeros((len(X),len(X)))
    for i in range(len(X)):
      temp = np.zeros(len(X))
      for j in range(len(X)):
          squared_norm = np.square(np.linalg.norm(X[i]-X[j],2))
          sum += squared_norm
          temp[j] = -1*squared_norm
      K[i] = temp
      #print(i)

    variance = sum/np.square(len(X))
    K = K/(2*variance)
    K = np.exp(K)


    return variance, K

def get_test_kernel(test_data,train_data,var):
    K_test = np.zeros((len(test_data),len(train_data)))
    for i in range(len(test_data)):
        temp = np.zeros(len(train_data))
        for j in range(len(train_data)):
            squared_norm = np.square(np.linalg.norm(test_data[i]-train_data[j]))
            temp[j] = np.exp(-1*squared_norm/(2*var))
        K_test[i] = temp
        #print(i)
    return K_test

def sigmoid(v):
    return torch.where(v >= 0,1 / (1 + torch.exp(-v)),torch.exp(v) / (1 + torch.exp(v)))

def dsigmoid(v):
    return sigmoid(v) * (1 - sigmoid(v))

def predict(w,inputs):
    return torch.matmul(torch.transpose(w,0,1),torch.transpose(inputs,0,1))
    
def cost_function(w,inputs,labels,c=1):
    """
    c = lambda.
    input must be a slice.
    """
    #w.grad=None
    #assert len(inputs) == len(labels), 'incompatible input and label length'
    y_hat = predict(w,inputs)
    v = torch.transpose(labels,0,1) * y_hat
    #print(v.shape)
    #cost = -torch.sum(torch.log(sigmoid(v))) + c*torch.matmul(torch.transpose(w,0,1),w)
    return -torch.sum(torch.log(sigmoid(v))) + c*torch.matmul(torch.transpose(w,0,1),w)

def get_accuracy(w,inputs,labels):
    v = predict(w,inputs)
    pred = sigmoid(predict(w,inputs))
    # print(pred)
    predicted_label = torch.where(pred>0.5,1,-1)
    # #print(pred.shape)
    count = 0
    for i in range(len(labels)):
        if pred[0][i] == labels[i]:
            count += 1
    
    return count/len(labels)

mat = scipy.io.loadmat('/ext3/Homework/data1.mat')

train_data = mat['TrainingX']
train_label = mat['TrainingY']
test_data = mat['TestX']
test_label = mat['TestY']

if os.path.isfile('./kernel_data/K_train.npy'):
    print ("File exist")
    K_train = np.load('./kernel_data/K_train')
else:
    var, K_train = get_variance_and_kernel(train_data)
    if not os.path.isdir('kernel_data'):
        os.mkdir('kernel_data')
    np.save('./kernel_data/K_train',K_train)

if os.path.isfile('./kernel_data/K_test.npy'):
    print ("File exist")
    K_test = np.load('./kernel_data/K_test')
else:
    K_test = get_test_kernel(test_data,train_data,var)
    if not os.path.isdir('kernel_data'):
        os.mkdir('kernel_data')
    np.save('./kernel_data/K_test',K_test)

X_train = torch.from_numpy(K_train).type(torch.FloatTensor)
y_train = torch.from_numpy(train_label).type(torch.FloatTensor)

X_test = torch.from_numpy(K_test).type(torch.FloatTensor)
y_test = torch.from_numpy(test_label).type(torch.FloatTensor)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

cost_log = []
w = torch.randn((10000,1),requires_grad=True,device=device)

lr = 1e-7
count = 0
while True:
    cost = cost_function(w,X_train,y_train,c=1)
    cost.backward()
    w.data -= lr*w.grad.data
    n = torch.norm(w.grad.data,2)
    cost_log.append(cost)
    w.grad.data.zero_()
    if count%50000 == 0:
        print(n)
    if n < 1e-5:
      break

torch.save(w,'w_gd.pt')
log_dict = {'cost_log': cost_log}
with open(f'cost_log_gd.json', 'w') as outfile:
    json.dump(log_dict, outfile)