# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:39:19 2019
@author: User
"""
import os
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy

#print(torch.cuda.current_device())
#print(torch.cuda.is_available())
 
np.random.seed(7)

class Model_task1(nn.Module):
    def __init__(self, image_channels=3, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels=image_channels,
                          out_channels=32,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.layer2 = nn.Sequential(
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.layer3 = nn.Sequential(
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.num_output_features = 128*4*4
        self.classifier = nn.Sequential(
                nn.Linear(self.num_output_features,64),
                nn.ReLU(),
                nn.Linear(64,num_classes),
                #nn.Softmax()
                )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        return x


class Model_task2_a(nn.Module):
    
    def __init__(self, image_channels=3, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                nn.Conv2d(in_channels=image_channels,
                          out_channels=32,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                #nn.BatchNorm2d(num_features = 32),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout(p=0),
                
                nn.Conv2d(in_channels=32,
                          out_channels=64,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                #nn.BatchNorm2d(num_features = 64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout(p=0),
                
                nn.Conv2d(in_channels=64,
                          out_channels=128,
                          kernel_size=5,
                          stride=1,
                          padding=2),
                #nn.BatchNorm2d(num_features = 128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout(p=0)
                )

        self.num_output_features = 128*4*4
        self.classifier = nn.Sequential(
                nn.Linear(self.num_output_features,64),
                nn.ReLU(),
                nn.Linear(64,num_classes),
                #nn.Softmax()
                )
        
        #self.feature_extractor.apply(self.init_weights)
        #self.classifier.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.01)
         
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        return x


class Trainer:

    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 30
        self.batch_size = 32
        self.learning_rate = 5e-2
        self.weight_decay = 0
        self.early_stop_count = 3

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        #self.model = Model_task1(image_channels=3, num_classes=10)
        self.model = Model_task2_a(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr = self.learning_rate,
                                         weight_decay = self.weight_decay)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train) // 2
        #print(self.validation_check)
        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc*100)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc*100)
        self.VALIDATION_LOSS.append(validation_loss)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc*100)
        self.TEST_LOSS.append(test_loss)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        self.validation_epoch()
        for epoch in range(self.epochs):
            print("Epoch " + str(epoch+1))
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(self.dataloader_train):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)    

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()
                
                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                 # Compute loss/accuracy for all three datasets.
                if (batch_it+1) % self.validation_check == 0:
                    self.validation_epoch()
                    # Check early stopping criteria.
                    if self.should_early_stop():
                        print("Early stopping.")
                        self.epochs = (epoch +1)
                        return

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()


def plot_metrics():
    """
    Plots the metrics and prints the final results.
    """
    os.makedirs("plots", exist_ok=True)
    
    fig1, ax1 = plt.subplots(1,1,figsize=(6,3.5),dpi=200)
    test_metric, = ax1.plot(trainer.TEST_LOSS, label="Testing Loss", color="red")
    validation_metric, = ax1.plot(trainer.VALIDATION_LOSS, label="Validation Loss", color="orange")
    training_metric, = ax1.plot(trainer.TRAIN_LOSS, label="Training Loss", color="blue")
    ax1.legend(handles=[training_metric,validation_metric,test_metric])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_xticks(np.linspace(0,len(trainer.TEST_LOSS)-1,6))
    ax1.set_xticklabels(np.round(np.linspace(0,trainer.epochs,6),5))
    plt.show()

    fig1.savefig(os.path.join("plots", "A3_T2_a_multiclass_crossentropy.eps"))

    fig2, ax1 = plt.subplots(1,1,figsize=(6,3.5),dpi=200)
    test_metric, = ax1.plot(trainer.TEST_ACC, label="Testing Accuracy", color="red")
    validation_metric, = ax1.plot(trainer.VALIDATION_ACC, label="Validation Accuracy", color="orange")
    train_metric, = ax1.plot(trainer.TRAIN_ACC, label="Training Accuracy", color="blue")
    ax1.legend(handles=[training_metric,validation_metric,test_metric])
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(np.linspace(0,len(trainer.TEST_LOSS)-1,6))
    ax1.set_xticklabels(np.round(np.linspace(0,trainer.epochs,6),5))
    plt.show()
    
    fig2.savefig(os.path.join("plots", "A3_T2_a_accuracy.eps"))
    
    print([param.nelement() for param in trainer.model.parameters()])
    print("Connections: " + str(sum([param.nelement() for param in trainer.model.parameters()][::2])))
    print("Biases: " + str(sum([param.nelement() for param in trainer.model.parameters()][1::2])))
    print("Trainable Parameters: " + str(sum([param.nelement() for param in trainer.model.parameters()])))

    print("Final training accuracy:", trainer.TRAIN_ACC[-trainer.early_stop_count])
    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])

plot_metrics()
