
# coding: utf-8

# In[40]:


# -*- coding: utf-8 -*-
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torchvision
import torchvision.transforms as transforms
import copy
import resnet as resNet
import numpy as np
import pdb
import matplotlib.pyplot as plt
import numpy as np
import pickle

alpha = 0.1
lambd = 0.01
use_adaptive = True
decreasing = False


# In[41]:


# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2000,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2000,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[42]:


# 2. Define a Convolution Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F


class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class simplerNet(nn.Module):
    def __init__(self):
        super(simplerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 11)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc3 = nn.Linear(3 * 5 * 5, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 3 * 5 * 5)
        x = self.fc3(x)
        return x
    
use_net = "resNet"

if use_net == "resNet":
    net = resNet.ResNet20()
elif use_net == "simplerNet":
    net = simplerNet()
else:
    net = simpleNet()

#use_cuda = torch.cuda.is_available()
use_cuda = True

if use_cuda:
    net = net.cuda()
    #net = nn.DataParallel(net).cuda()


# In[43]:


# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

use_optim = "adaptive"

criterion = nn.CrossEntropyLoss()

if use_optim == "adagrad":
    optimizer = optim.Adagrad(net.parameters(), lr=alpha)
elif use_optim == "adam":
    optimizer = optim.Adam(net.parameters(), lr=alpha)
else:
    optimizer = optim.SGD(net.parameters(), lr=alpha, momentum=0.0)


# In[44]:


def printWeights(net):    
    for currName, currValue, in net.named_parameters():
#         if currName == "conv1.bias":
        print(currName)
        print(currValue.data) 


# In[45]:


#Copy of the network, creates XBar network initialized with weights and biases = 0
def zeroNet(net):
    zeroNet = deepCopyWithGradients(net)
    for paramName, paramValue, in zeroNet.named_parameters():
        paramValue.data = paramValue.data.new_zeros(paramValue.data.cpu().size())
        paramValue.grad = paramValue.grad.new_zeros(paramValue.grad.data.cpu().size())
        
    return zeroNet


# In[46]:


def netMeanGrad(X):
    meanGrad = zeroNet(X[0])
    for network in X:
        for paramName, paramValue, in network.named_parameters():
            for meanGradName, meanGradValue in meanGrad.named_parameters():
                if paramName == meanGradName:
                    meanGradValue.grad += paramValue.grad
    
    for meanGradName, meanGradValue in meanGrad.named_parameters():
        meanGradValue.grad /= np.size(X)
    
    return meanGrad


# In[47]:


def deepCopyWithGradients(net):
    #call deepcopy
    netCopy = copy.deepcopy(net)
    # create optimizer and attach it to netCopy
    optimizerCopy = optim.SGD(netCopy.parameters(), lr=alpha, momentum=0.0)
    optimizerCopy.zero_grad()
    # Copy the values of the gradients by looping over parmeters
    for paramName, paramValue, in net.named_parameters():
        for netCopyName, netCopyValue, in netCopy.named_parameters():
            if paramName == netCopyName:
                netCopyValue.grad = paramValue.grad.clone()
    return netCopy


# In[48]:


def L2normNetGradients(net):
    gradNorm = 0
    for paramName, paramValue, in net.named_parameters():
        paramNorm = torch.pow(torch.norm(paramValue.grad.data, 2),2)
        gradNorm += paramNorm
    gradNorm = np.sqrt(gradNorm)
    return gradNorm


# In[49]:


def netDivide(net, value):
    netCopy = deepCopyWithGradients(net)
    for paramName, paramValue, in netCopy.named_parameters():
        if use_cuda:
            paramValue.grad /= value.cuda()
        else:
            paramValue.grad /= value
    return netCopy


# In[50]:


reduction = 2
def adjust_learning_rate(optimizer, reduction, originalAlpha):
    lr = originalAlpha/reduction
    for param_group in optimizer.param_groups:
         param_group['lr'] = lr
    return lr


# In[51]:


def regularizeNet(net, preCopy, alpha, lambd):
    netCopy = deepCopyWithGradients(net)
    for paramName, paramValue, in netCopy.named_parameters():
        for preCopyName, preCopyValue, in preCopy.named_parameters():
            if paramName == preCopyName:
                paramValue.data = paramValue.data - alpha*lambd*torch.sign(preCopyValue.data)
    return netCopy


# In[52]:


def regCost(preCopy, lambd):
    temp = 0
    for preCopyName, preCopyValue, in preCopy.named_parameters():
        temp += lambd*torch.norm(preCopyValue.data, 1)
    return temp


# In[53]:


# 4. Train the network

n_traj_hist, threshold = 20, 0.5
iterationIndices, objectiveValues = [], []
iteration = 0
alphaCount, prevAlpha, originalAlpha = 0, alpha, alpha
markAlphaIter, markAlpha = [], []
alphaIteration, alphaValues = [], []
gradientSet = []
avgSearchDirectionSizes,avgSearchDirectionIter = [],[]
searchVector_unit = []

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 1/reduction, last_epoch=-1)
for epoch in range(40):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer = optim.SGD( net.parameters(), lr=alpha, momentum=0.0 )
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        preUpdateCopy = copy.deepcopy(net)
        loss.backward()
        optimizer.step()

        net = regularizeNet(net, preUpdateCopy, alpha, lambd)
        
        # print statistics
        printEvery = 25
        running_loss += loss.item()
        if iteration % printEvery == printEvery-1:
            print('[%7d, %d, %5d] loss: %.3f' %
                  (iteration+1, epoch + 1, i + 1, (running_loss + regCost(preUpdateCopy, lambd))/ printEvery))
            running_loss = 0.0   
        
        #value vs iteration 
        objectiveValues.append(loss)
        iterationIndices.append(iteration)
        alphaValues.append(alpha)
        
        if decreasing:
            alpha = adjust_learning_rate(optimizer, reduction, originalAlpha)
            reduction += 1
        
        #searchVectors and adaptive learning rate
        gradientSet.append( deepCopyWithGradients(net) )
        if iteration >= n_traj_hist-1 and np.size(gradientSet) == n_traj_hist:
            if alpha == prevAlpha:
                alphaCount += 1
            else:
                alphaCount = 1
                
            searchVector_unit = [netDivide(gradient,L2normNetGradients(gradient)) for gradient in gradientSet]
            
            avg_search = netMeanGrad(searchVector_unit)
            avg_search_norm = L2normNetGradients(avg_search)
            avgSearchDirectionSizes.append(avg_search_norm)
            avgSearchDirectionIter.append(iteration)
                    
            if use_adaptive:
                prevAlpha = alpha
                #print(avg_search_norm,threshold)
                if avg_search_norm < threshold or alphaCount == 10000:
                    alpha = adjust_learning_rate(optimizer, reduction, originalAlpha)
                    reduction += 1
                    print("Iteration: " + str(iteration) + " Alpha: " + str(alpha))
                    markAlpha.append(loss)
                    markAlphaIter.append(iteration) 
                        
                    gradientSet = []
            gradientSet = gradientSet[1:]
        
        iteration += 1
        
print('Finished Training')


# In[54]:


def plot(axis_x, axis_y, alpha, plot_type = 'Value', showAlpha = False):
    axis_x = axis_x[0:]
    axis_y = axis_y[0:]
#     plt.ylim(1.81,2.58)
    
    if plot_type == 'Relative Error':
        plt.semilogy(axis_x, axis_y, color = 'k', linestyle = '-')
    else:
        plt.plot(axis_x, axis_y, color='k', linestyle='-')
    plt.title(plot_type + ' vs Iteration ', fontsize=15)# + r'$\alpha$ = ' + str(alpha), fontsize=15)
    plt.xlabel('iterations')
    plt.ylabel(plot_type)
    if showAlpha:
        plt.plot(markAlphaIter, markAlpha, color='r', marker='x', linestyle="None")
    plt.show()

relativeObjectiveValues = abs(np.subtract(objectiveValues,1.5))/abs(1.5)
    
plot(avgSearchDirectionIter, avgSearchDirectionSizes, alpha, plot_type = "Avg Search Direction", showAlpha = False)
plot(iterationIndices, objectiveValues, alpha, plot_type = 'Value', showAlpha = True)
plot(iterationIndices, relativeObjectiveValues, alpha, plot_type = 'Relative Error')
plot(iterationIndices, alphaValues, alpha, plot_type = 'Alpha', showAlpha = False)
#printWeights(net)


# In[55]:


saveAvgSearchDirectionSizesHist20 = avgSearchDirectionSizes
saveObjectiveValuesHist20 = objectiveValues
saveRelativeErrorHist20 = relativeObjectiveValues
saveAlphaHist20 = alphaValues
plot(iterationIndices, saveObjectiveValuesHist20, alpha, plot_type = 'Value')

# with open('objsFixed.pkl', 'wb') as f: 
#     pickle.dump([saveAvgSearchDirectionSizes, saveObjectiveValues, saveRelativeError, saveAlpha], f)
    
# with open('objsDecreasing.pkl', 'wb') as k:
#     pickle.dump([saveAvgSearchDirectionSizes2, saveObjectiveValues2, saveRelativeError2, saveAlpha2], k)
    
with open('objsAdaptive.pkl', 'wb') as p:
    pickle.dump([saveAvgSearchDirectionSizes3, saveObjectiveValues3, saveRelativeError3, saveAlpha3], p)
# In[56]:


with open('adaptiveHist20.pkl', 'wb') as f:
    pickle.dump([saveAvgSearchDirectionSizesHist20, saveObjectiveValuesHist20, saveRelativeErrorHist20, saveAlphaHist20],f)


# In[63]:


with open('objsFixed.pkl', 'rb') as f:
    saveAvgSearchDirectionSizes, saveObjectiveValues, saveRelativeError, saveAlpha = pickle.load(f)
    
with open('objsDecreasing.pkl', 'rb') as k:
    saveAvgSearchDirectionSizes2, saveObjectiveValues2, saveRelativeError2, saveAlpha2 = pickle.load(k)
    
with open('objsAdaGrad.pkl', 'rb') as p:
    saveAvgSearchDirectionSizes3, saveObjectiveValues3, saveRelativeError3, saveAlpha3 = pickle.load(p)

with open('objsAdaptive.pkl', 'rb') as s:
    saveAvgSearchDirectionSizes4, saveObjectiveValues4, saveRelativeError4, saveAlpha4 = pickle.load(s)
    
with open('adaptiveHist5.pkl', 'rb') as r:
    saveAvgSearchDirectionSizesHist5, saveObjectiveValuesHist5, saveRelativeErrorHist5, saveAlphaHist5 = pickle.load(r)

with open('adaptiveHist10.pkl', 'rb') as z:
    saveAvgSearchDirectionSizesHist10, saveObjectiveValuesHist10, saveRelativeErrorHist10, saveAlphaHist10 = pickle.load(z)
    
with open('adaptiveHist20.pkl', 'rb') as m:
    saveAvgSearchDirectionSizesHist20, saveObjectiveValuesHist20, saveRelativeErrorHist20, saveAlphaHist20 = pickle.load(m)
    


# In[65]:


iterationIndices = [x for x in range(1000)]

plt.close('all')
plt.figure(1)
plt.semilogy(iterationIndices, saveAlpha, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveAlpha2, linestyle ='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveAlpha3, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveAlpha4, linestyle = '-', alpha = 0.8)
plt.title('Alpha vs Iteration ' + r'$\alpha$ = ' + str(saveAlpha[0]), fontsize=15)
plt.xlabel('Iterations')
plt.ylabel('Alpha')
plt.legend(['Fixed', 'Decreasing','AdaGrad','Adaptive'])


plt.figure(2)
plt.semilogy(iterationIndices, saveRelativeError, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveRelativeError2, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveRelativeError3, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveRelativeError4, linestyle='-', alpha = 0.8)
plt.title('Relative Error vs Iteration ' + r'$\alpha$ = ' + str(saveAlpha[0]), fontsize=15)
plt.xlabel('Iterations')
plt.ylabel('Relative Error')
plt.legend(['Fixed', 'Decreasing','AdaGrad','Adaptive'])

plt.figure(3)
plt.semilogy(iterationIndices, saveRelativeErrorHist5, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveRelativeErrorHist10, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveRelativeErrorHist20, linestyle='-', alpha = 0.8)
plt.title('Relative Error vs Iteration ' + r'$\alpha$ = ' + str(saveAlpha[0]), fontsize=15)
plt.xlabel('Iterations')
plt.ylabel('Relative Error')
plt.legend(['histSize=5','histSize=10','histSize=20'])

plt.figure(4)
plt.semilogy(iterationIndices, saveAlphaHist5, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveAlphaHist10, linestyle='-', alpha = 0.8)
plt.semilogy(iterationIndices, saveAlphaHist20, linestyle='-', alpha = 0.8)
plt.title('Alpha vs Iteration ' + r'$\alpha$ = ' + str(saveAlpha[0]), fontsize=15)
plt.xlabel('Iterations')
plt.ylabel('Alpha')
plt.legend(['histSize=5','histSize=10','histSize=20'])

plt.show()


# In[59]:


# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[60]:


# Okay, now let us see what the neural network thinks these examples above are:

if use_cuda:
    outputs = net(images.cuda())
else:
    outputs = net(images)


########################################################################
# The outputs are energies for the 10 classes.
# Higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# In[61]:


# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[62]:


# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

