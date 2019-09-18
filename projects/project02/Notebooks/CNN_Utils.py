import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class CNN_Utils():
    
    def __init__(self, my_lr, batch_size, epochs):
        
        self.learning_rate = my_lr
        self.batch_size = batch_size
        self.epochs = epochs

    def get_error(self, scores , labels ):
    
        bs=scores.size(0)
        predicted_labels = scores.argmax(dim=1)
        
        indicator = (predicted_labels == labels)
        num_matches=indicator.sum()
        
        return 1-num_matches.float()/bs   
        
    
    def prep_train_validate_data(self, train_images, train_labels, split=0.00):
    
        #### here 'test' actually means validation
        x_train, x_validate, y_train, y_validate = train_test_split(train_images, train_labels, test_size = split)
        
        ### Normalize ##
        x_train = x_train / 255
        x_validate = x_validate / 255
    
        ##################################################################################################
        
        
        x_train = torch.from_numpy(x_train)
        x_validate = torch.from_numpy(x_validate)
        y_train = torch.from_numpy(y_train)
        y_validate = torch.from_numpy(y_validate)
    
        return x_train, x_validate, y_train, y_validate


    def prep_test_data(self, test_images, test_labels):
        
        x_test = test_images
        x_test = x_test / 255
    
        x_test = torch.from_numpy(x_test)
        y_test = torch.from_numpy(test_labels)
        
        return x_test, y_test
        
    
    
    def test_Cnet(self, net, test_data, test_lbls, device, test_history=None):
    
        running_error=0
        running_loss=0
        num_batches=0
    
        criterion = nn.CrossEntropyLoss()
    
        for i in range(0, len(test_data), self.batch_size):
    
            minibatch_data =  test_data[i:i+self.batch_size].unsqueeze(dim=1).cuda()
            minibatch_label = test_lbls[i:i+self.batch_size]
    
            minibatch_data=minibatch_data.to(device)
            minibatch_label=minibatch_label.to(device)
            
            inputs = minibatch_data
    
            scores=net( inputs ) 
            
            loss =  criterion( scores , minibatch_label )
            
            running_loss += loss.item()
    
            error = self.get_error( scores , minibatch_label)
    
            running_error += error.item()
    
            num_batches+=1
    
            
        total_loss = running_loss/num_batches
        total_error = running_error/num_batches
        
        if test_history!=None:
            test_history[1].append(total_loss)
            test_history[2].append(total_error*100)
            
            
        
        return total_error*100
        
        
        
    def train_Cnet(self, net, train_data, train_lbls, val_data, val_lbls, device):
        
        lr = self.learning_rate
        
        train_history = [[], [], []]
        test_history = [[], [], []]
        
        criterion = nn.CrossEntropyLoss()
    
        start=time.time()
    
        for epoch in range(1,self.epochs):
            
            if not epoch%10:
                lr = lr / 1.5
                
            optimizer=torch.optim.Adam( net.parameters() , lr=lr )
                
            running_loss=0
            running_error=0
            num_batches=0
            
            trN = len(train_data)
            
            shuffled_indices=torch.randperm(trN)
        
            for count in range(0,trN, self.batch_size):
                
                # FORWARD AND BACKWARD PASS
            
                optimizer.zero_grad()
                    
                indices = shuffled_indices[count:count + self.batch_size]
                
                minibatch_data =  train_data[indices].unsqueeze(dim=1).cuda()
                minibatch_label=  train_lbls[indices]
                
                minibatch_data=minibatch_data.to(device)
                minibatch_label=minibatch_label.to(device)
    
                inputs = minibatch_data
        
                inputs.requires_grad_()
    
                scores=net( inputs ) 
        
                minibatch_label = minibatch_label.long()
                
                loss =  criterion( scores , minibatch_label )           
    
                loss.backward()
                
                optimizer.step()
                
        
                # COMPUTE STATS
                
                running_loss += loss.detach().item()
                
                error = self.get_error( scores.detach() , minibatch_label)
    
                running_error += error.item()
                
                num_batches+=1        
            
            
            # AVERAGE STATS THEN DISPLAY
            total_loss = running_loss/num_batches
            total_error = running_error/num_batches
            elapsed = (time.time()-start)/60
            
                    
            train_history[0].append(epoch)
            train_history[1].append(total_loss)
            train_history[2].append(total_error*100)
            
            test_history[0].append(epoch)
            
            #print('epoch=',epoch, '\t time=', elapsed,'min', '\t lr=', lr  ,'\t loss=', total_loss , '\t error=', total_error*100 ,'percent')
    
    
            self.test_Cnet(net.eval(), val_data, val_lbls, device, test_history)
    
            
        #print( '\nElapsed time=', elapsed, 'min')   
        return train_history, test_history
    

        
    def plot_CV_history(self, train_history_over_CV, val_history_over_CV):
               
        ln = len(train_history_over_CV)
        for i in range(1,ln):
                for x in range(len(train_history_over_CV[0][1])):
                    train_history_over_CV[0][1][x] += train_history_over_CV[i][1][x]
                    train_history_over_CV[0][2][x] += train_history_over_CV[i][2][x]
                    
                    if i == ln-1:
                        train_history_over_CV[0][1][x] /= ln
                        train_history_over_CV[0][2][x] /= ln
        
        ln = len(val_history_over_CV)
        for i in range(1,ln):
                for x in range(len(val_history_over_CV[0][1])):
                    val_history_over_CV[0][1][x] += val_history_over_CV[i][1][x]
                    val_history_over_CV[0][2][x] += val_history_over_CV[i][2][x]
                    
                    if i == ln-1:
                        val_history_over_CV[0][1][x] /= ln
                        val_history_over_CV[0][2][x] /= ln
        
        
        
        plt.figure(figsize=(18, 6))
        plt.subplot(1,2,1)
        plt.plot(train_history_over_CV[0][0], train_history_over_CV[0][1])
        plt.plot(val_history_over_CV[0][0], val_history_over_CV[0][1])
        plt.legend(['Average Train Loss', ' AverageValidation Loss'], loc='upper right', prop={'size': 15})
        
        plt.subplot(1,2,2)
        plt.plot(train_history_over_CV[0][0], train_history_over_CV[0][2])
        plt.plot(val_history_over_CV[0][0], val_history_over_CV[0][2])
        plt.legend(['Average Train Error', 'Average Validation Error'], loc='upper right', prop={'size': 15})
        plt.yticks(np.arange(0, 100, step=5))
        
        plt.suptitle('Cross Validation Performance', fontsize=15.0, y=1.08, fontweight='bold')
         

        
        
    def plot_FullTrain_history(self, train_history, test_history):
         
        
        plt.figure(figsize=(18, 6))
        plt.subplot(1,2,1)
        plt.plot(train_history[0], train_history[1])
        plt.plot(test_history[0], test_history[1])
        plt.legend(['Train Loss', 'Test Loss'], loc='upper right', prop={'size': 15})
        
        plt.subplot(1,2,2)
        plt.plot(train_history[0], train_history[2])
        plt.plot(test_history[0], test_history[2])
        plt.legend(['Train Error', 'Test Error'], loc='upper right', prop={'size': 15})
        plt.yticks(np.arange(0, 100, step=5))

        plt.suptitle('Training on entire Training Data, with Peformance on Test Data', fontsize=15.0, y=1.08, fontweight='bold')
        