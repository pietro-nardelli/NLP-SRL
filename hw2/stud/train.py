import torch
import torch.optim as optim

#TQDM is a A Fast, Extensible Progress Bar for Python and CLI https://tqdm.github.io
from tqdm import tqdm
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

from stud.my_model import *

#Class to train and evaluate the model
class Trainer():

    def __init__(
        self,
        device,
        model,
        loss_function,
        optimizer,
        model_path
        ):

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.model_path = model_path


    #Plot train and validation loss
    def plotLearning (self, N:int, train_loss_list:List[str], val_loss_list:List[str]):
        x = [i for i in range(N)]

        plt.ylabel('Loss')
        plt.xlabel('Epochs')


        plt.plot(x, train_loss_list, 'b-', label='train_loss')
        plt.plot(x, val_loss_list, 'c-', label='val_loss')
        plt.legend(loc="upper right")
        plt.plot()
        #plt.savefig(filename)

    # The train method prints the average training loss on train_dataset over epochs. 
    # print_step = True would print the avg_train loss for each batch
    def train(self, train_dataset, valid_dataset, train_words:List[List[str]], dev_words:List[List[str]], epochs=1, print_step=False):
        
        train_loss_list = []
        val_loss_list = []
        models_state = []
        

        for epoch in tqdm(range(epochs)):  
            # Just for TQDM to avoid multiple printing
            time.sleep(0.1)
            epoch_loss = 0

            # Train mode: with dropout
            self.model.train() 
            self.model.to(self.device)

            # Counter to take in memory the last sentence processed
            k = 0
            for step, sample in enumerate(train_dataset):
                word_inputs = sample['word_inputs']
                predicates_flags_inputs = sample['predicates_flags_inputs']
                lemmas_inputs = sample['lemmas_inputs']
                pos_tags_inputs = sample['pos_tags_inputs']

                # The BERT inputs should be appended and passed to the model to create BERT batches.
                # This is doing  here (instead of inside the dataloader), because the GPU memory is not so large
                bert_inputs = []
                for i in range(word_inputs.shape[0]):
                  bert_inputs.append(train_words[k])
                  k += 1
                freq_inputs = sample['freq_inputs']
                labels = sample['outputs']
                    

                # we need to set the gradients to zero before starting to do backpropragation
                # because PyTorch accumulates the gradients on subsequent backward passes
                self.optimizer.zero_grad()

                predictions = self.model(word_inputs, predicates_flags_inputs, lemmas_inputs, pos_tags_inputs, bert_inputs, freq_inputs, word_drop=HParam.word_drop)
                loss = self.loss_function(predictions, labels.view(-1).long())
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                # Print at each step the epoch loss
                if (print_step):
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, epoch_loss/(step+1) ))


            avg_epoch_loss = epoch_loss / len(train_dataset)
            val_loss= self.evaluate(valid_dataset, dev_words)

            # Instantiated for plotLearning
            train_loss_list.append(avg_epoch_loss)
            val_loss_list.append(val_loss)

            print ("\nepoch: ", epoch+1, "avg_epoch_loss: ", avg_epoch_loss,"val_loss: ", val_loss)
            
            # Save the model at each epoch 
            models_state.append(self.model.state_dict())
            if (epoch != 0):
                now = datetime.now() # current date and time
                dt = now.strftime("%d-%m-%Y_%H-%M-%S")

                torch.save(models_state[epoch], self.model_path+"model_"+dt+"_epoch_"+str(epoch+1)+".pt")
                print ("Model for the epoch "+str(epoch+1)+" saved.")

        print('Finished Training')
        self.plotLearning(epochs, train_loss_list, val_loss_list)

    # The evaluation method prints the average validation loss over valid_dataset.
    def evaluate(self, valid_dataset, dev_words:List[List[str]]):
        valid_loss = 0.0

        self.model.eval() #Evaluation mode: no dropout
        self.model.to(self.device)

        # no gradient updates here
        with torch.no_grad():
            k = 0
            for sample in valid_dataset:
                word_inputs = sample['word_inputs']
                predicates_flags_inputs = sample['predicates_flags_inputs']
                lemmas_inputs = sample['lemmas_inputs']
                pos_tags_inputs = sample['pos_tags_inputs']

                # (Check training method for further information about bert_inputs)
                bert_inputs = []
                for i in range(word_inputs.shape[0]):
                  bert_inputs.append(dev_words[k])
                  k += 1

                freq_inputs = sample['freq_inputs']
                labels = sample['outputs']
                predictions = self.model(word_inputs, predicates_flags_inputs, lemmas_inputs, pos_tags_inputs, bert_inputs, freq_inputs).to(self.device)


                loss = self.loss_function(predictions, labels.view(-1).long())
                valid_loss += loss.item()

        return valid_loss / len(valid_dataset)
