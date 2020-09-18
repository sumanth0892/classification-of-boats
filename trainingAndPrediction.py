from torchvision import models
import torch.nn as nn
from torch import optim
import dataPrep as dP
import torch 

model = models.vgg11(pretrained = True)
#Freeze the model weights
for param in model.parameters():
	param.requires_grad = False 
#print(model)

n_classes = 9
#n_inputs = model.fc.in_features
model.classifier = nn.Sequential(
			nn.Linear(25088,256),
			nn.ReLU(),
			nn.Dropout(0.4),
			nn.Linear(256,n_classes),
			nn.Softmax(dim = 1))
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr = 0.003,momentum = 0.9)
trainloader,testloader = dP.getData()

def train(trainLoader,testLoader,nEpochs,model = model,loss_fn = loss_fn,optimizer = optimizer):
	#Train the model
	losses = [] 
	for epoch in range(1,nEpochs+1):
		training_loss = 0.0
		for images,labels in trainLoader:
			optimizer.zero_grad()
			outputs = model(images)
			loss = loss_fn(outputs,labels)
			loss.backward()
			optimizer.step()
			training_loss += loss.item()
		if epoch%5 == 0:
			print(training_loss/len(trainLoader))
		losses.append(training_loss/len(trainLoader))
	correctPredicted = 0; totalPredicted = 0
	for images,labels in testLoader:
		for i in range(len(labels)):
			img = images[i]
			c,w,h = img.shape
			img = img.view((1,c,w,h))
			true_label = labels.numpy()[i]
			with torch.no_grad():
				output = model(img)
			_,pred_label = torch.max(output.data,1)
			if true_label == pred_label:
				correctPredicted += 1
			totalPredicted += 1
	print("Training Losses are:")
	print(losses)
	print("\n")
	print("Testing accuracy:")
	print(correctPredicted*100/totalPredicted)

train(trainloader,testloader,10,model,loss_fn,optimizer)