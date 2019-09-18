import matplotlib.pyplot as plt
import numpy as np
import copy
import string
import csv
import random
from skimage import transform as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from global_defs import *


# LSTM
hidden_size = 256
vocab_size = 26

SOW_token = 0


# CLSTM
hidden_dim_of_lstm1 = 256
bi_dir = True

        
def csv2images(fileStr):
    dataStr = csv.reader(open(fileStr), delimiter='\n', quotechar='|')
    data = []
    next(dataStr)
    for row in dataStr:
        eachRow = ','.join(row);
        rowArr = list(map(int, eachRow.split(',')));
        data.append(rowArr)
    
    return data

alphas = list(string.ascii_lowercase)



def get_data(wordImageIndices, images, size):
    
    data = []  
    labels = []
    
    for wim in wordImageIndices[:size]:

        word_imgs = torch.tensor( [ images[i][1:] for i in wim[1] ] )  ###fetch image data in sequence
        word_imgs = word_imgs.reshape(word_imgs.shape[0], 28, 28).float()  ### reshape to 28 x 28 images in sequence
        word_imgs /= 255            #### make float!!

        img_labels = [ images[i][0] for i in wim[1] ] 

        data.append(word_imgs)
        labels.append(torch.tensor(img_labels))

    return data, labels



def get_unique_words(labels, test_data_size):
    
    words = []
    for w in labels[:test_data_size]:
        word = [alphas[x] for x in w]
        words.append(''.join(word))

    words = np.array(words)

    uniq = np.unique(words)

    print('No. of Unique Words: ', len(uniq), '\n')
    #print(np.unique(words))
    
        
        
        
def plot_prediction_pie(errors, model_names):
    
    print('\n')

    figcolor = '1.0'
    plt.figure(figsize=(14, 8), facecolor = figcolor)

    
    for i in range(len(model_names)):
        
        plt.subplot(2,3,i+1)
        
        err = errors[i]
        
        lbls = ['Correct', 'Incorrect']
        sizes = [100-err, err]
        explode = (0.1, 0.1)

        patches, texts, autotexts = plt.pie(sizes, explode=explode, labels=lbls, autopct='%1.1f%%', colors=['green', 'magenta'],
            shadow=True, startangle=90, labeldistance=1.1)

        for t in range(len(texts)):
            texts[t].set_fontsize(10)

        for at in range(len(autotexts)):
            autotexts[at].set_fontsize(13)
            autotexts[at].set_weight('bold')
            autotexts[at].set_color('white')
        
        plt.title(model_names[i], fontsize=15.0, fontweight='bold')
    
    plt.suptitle('Breakdown of Prediction', fontsize=20.0, y=1.08 , color='black', fontweight='bold')
    
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5, top = 0.9 )


def plot_word_mismatch(mismatches, model_names):
    
    print('\n')

    figcolor = '1.0'
    plt.figure(figsize=(14, 8), facecolor = figcolor)

    
    for i in range(len(model_names)):
        
        plt.subplot(2,3,i+1)
        
        mismatch_cnt = mismatches[i]
        
        rng = 11
        mismatch_cnt = copy.copy(mismatch_cnt[:rng-1])
        
        
        sm = 0
        cnt = 0
        for j in range(len(mismatch_cnt)):
            sm += (j+1) * mismatch_cnt[j]
            cnt += mismatch_cnt[j]
        avg_mis = sm / cnt

        
        plt.bar(range(1,rng), mismatch_cnt, align='center')
        plt.xticks(np.arange(1, rng, step=1))

        plt.xlabel('Count of Wrong Alphabets\n\n' + r'$\bf{' + 'Average Count: ' + str(avg_mis) + '}$', fontsize=15.0)
        plt.ylabel('No. of Predicted words\n with this count', fontsize=13.0)
        
        plt.title(model_names[i], y=1.2, fontsize=15.0, fontweight='bold')
        
    
    plt.suptitle('Predicted Word v/s True Word\n Mismatch as count of Wrong Alphabets', fontsize=20.0, 
                                                                         y=1.15 , color='black', fontweight='bold')
    
    plt.subplots_adjust(wspace = 0.5, hspace = 1.2, top = 0.9 )

    
    
def plot_post_processor_stats(postproc_stat, model_name=None):
    
    print('\n')
    
    m = model_name.split()
    input_model = m[len(m)-1]
    output_model = m[0]
    
    figcolor = '1.0'
    plt.figure(figsize=(18,2), facecolor = figcolor, frameon=False)
        
    plt.subplot(1,1,1, frame_on=False)
    
    postproc_stat = np.array(postproc_stat)
    data = copy.copy(postproc_stat.transpose())

    h = len(data)
    w = len(data[0])
        
    colLabels=['0']
    for x in range(1,w-1):
        colLabels.append( str(x) ) 
    colLabels.append('>='+ str(x) )
        
    tb = plt.table(cellText=data, loc=(0,0), cellLoc='center', bbox=None,
         colLabels=colLabels, 
         rowLabels=['To Correct outputs from '+ output_model, 'To Incorrect outputs from '+ output_model],
    )
        
    tc = tb.properties()['child_artists']
    for cell in tc: 
        cell.set_height(1/h)
        cell.set_width(1/(w+1))
        
    tb.auto_set_font_size(False)
    tb.set_fontsize(15)
        
    plt.xticks([])
    plt.yticks([])
        
    plt.title(model_name + '\n\nCount of Wrong Alphabets in Inputs from ' + input_model, 
              fontsize=20.0, y=1.5 , color='black', fontweight='bold')
    
    
    
    
    
    
    
    
    
#############################################  Evaluation Functions ###################################################    

def BasicCNN_evaluation(data, labels, model):
    
    test_data = copy.deepcopy(data)
    test_labels = copy.deepcopy(labels)

    num_words = len(test_data)

    mismatch_cnt = np.zeros(20, dtype=int) #### we know that no. of characters in max. lenght word is 19
    
    spellings = []
    
    err = 0
    
    for wrd in range(num_words):
        
        input_tensor = test_data[wrd].unsqueeze(dim=1).double()
        target_tensor= test_labels[wrd]
        
        if device.type == "cuda":
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
        
        scores = model( input_tensor )

        chars = scores.argmax(dim=1)   #### the output is an array of alphabet label (numbers) for single word

        out_word = [alphas[x] for x in chars]  #### convert labels to alphabets
        out_word = ''.join(out_word)
        
        tar_word = [alphas[x] for x in target_tensor]
        tar_word = ''.join(tar_word)

        if tar_word != out_word:
            err += 1
            
            mis_cnt = sum(a!=b for a, b in zip(tar_word, out_word))
            mismatch_cnt[mis_cnt-1] += 1
            
                
        spellings.append(chars)

        
    err = err/num_words * 100
    
    return spellings, mismatch_cnt, err
    
    

def LSTM_PostProc_evaluation(data, labels, model):
    
    test_data = copy.deepcopy(data)
    test_labels = copy.deepcopy(labels)
        
        
    num_words = len(test_data)

    err = 0
    in_out_stats = np.zeros((6,2))
    
    mismatch_cnt = np.zeros(20, dtype=int)
    spellings = []
    
    for wrd in range(num_words):
   
        input_tensor = test_data[wrd]
        target_tensor= test_labels[wrd]

        if device.type == "cuda":
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
        
        # input and target words length
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        
        input_tensor += 1
        
        #initial hidden states
        h = torch.zeros(2, 1, hidden_size) ### if bi_dir=True, first dimension is 2 for bi-directional
        c = torch.zeros(2, 1, hidden_size)
        h = h.to(device)
        c = c.to(device)

        out, h, c = model(input_tensor.view(input_length,1), h, c)      
        
        in_word = [alphas[x-1] for x in input_tensor]
        in_word = ''.join(in_word)
        
        
        chars = out.argmax(dim=2)
        out_word = [alphas[x-1] for x in chars]  
        out_word = ''.join(out_word)
        #print(in_word, out_word)

        tar_word = [alphas[x] for x in target_tensor]
        tar_word = ''.join(tar_word)
        

        if in_word==tar_word and out_word==tar_word:  ### making correct->correct
            #print(in_word, out_word)
            in_out_stats[0][0] +=1
        
                    
        if in_word==tar_word and out_word!=tar_word:  ### making correct->incorrect
            #print(wrd, in_word, out_word)
            in_out_stats[0][1] +=1
            
            
        if in_word!=tar_word and out_word==tar_word: ### making incorrect->correct
            #print(in_word, out_word)
            mct = sum(a!=b for a, b in zip(tar_word, in_word))
            if mct <= 4:
                in_out_stats[mct][0] +=1
            else:
                in_out_stats[5][0] +=1


        if in_word!=tar_word and out_word!=tar_word:  ### incorrect -> incorrect
            #print(in_word, out_word)
            mct = sum(a!=b for a, b in zip(tar_word, in_word))
            if mct <= 4:
                in_out_stats[mct][1] +=1
            else:
                in_out_stats[5][1] +=1
        

        
        mis_cnt = sum(a!=b for a, b in zip(tar_word, out_word))
        if mis_cnt > 0:
            err += 1
            mismatch_cnt[mis_cnt-1] += 1
            
        
 
        spellings.append(chars-1)
        


    err = err/num_words * 100
    
    #plt.matshow(cnt)

    return spellings, mismatch_cnt, err, in_out_stats
    
    

def Seq2Seq_PostProc_evaluation(data, labels, encoder, decoder):

    
    test_data = copy.deepcopy(data)
    test_labels = copy.deepcopy(labels)
        
        
    num_words = len(test_data)

    err = 0
    in_out_stats = np.zeros((6,2))
    
    mismatch_cnt = np.zeros(20, dtype=int)
    spellings = []
    
    for wrd in range(num_words):

        input_tensor = test_data[wrd]
        target_tensor= test_labels[wrd]
        
        
        if device.type == "cuda":
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
        
        input_tensor += 1


        # input and target words length
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        

        #initial hidden states
        encoder_h = torch.zeros(1, 1, hidden_size)
        encoder_c = torch.zeros(1, 1, hidden_size)

        #send to GPU
        encoder_h=encoder_h.to(device)
        encoder_c=encoder_c.to(device)

        #encoding
        encoder_out,encoder_h,encoder_c = encoder(input_tensor.view(input_length,1), encoder_h, encoder_c)

        # Delivering output feature vector of the encoder into the decoder 
        # applying output hidden feature of encoder as input hidden feature to decoder
        decoder_h = encoder_h 
        decoder_c = encoder_c

        #initial input decoder
        decoder_input=torch.LongTensor([SOW_token])
        if device.type == "cuda":
            decoder_input = decoder_input.cuda()
        # transfering feature vector from encoder to decoder
        decoder_h = encoder_h 
        decoder_c = encoder_c
        # decoding
        # outputs t ozero
        
        out = torch.zeros(input_length, vocab_size).to(device)

       
        for dc in range(input_length):
            decoder_out, deocder_h, decoder_c= decoder(decoder_input.view(1,1),decoder_h,decoder_c)

            out[dc]=decoder_out
            
            top1 = decoder_out[0][0].argmax()
            decoder_input=top1
            
            
            
            
        in_word = [alphas[x-1] for x in input_tensor]
        in_word = ''.join(in_word)
        
        
        chars = out.argmax(dim=1)
        out_word = [alphas[x-1] for x in chars]  
        out_word = ''.join(out_word)
        #print(in_word, out_word)

        tar_word = [alphas[x] for x in target_tensor]
        tar_word = ''.join(tar_word)
        

        if in_word==tar_word and out_word==tar_word:  ### making correct->correct
            #print(in_word, out_word)
            in_out_stats[0][0] +=1
        
                    
        if in_word==tar_word and out_word!=tar_word:  ### making correct->incorrect
            #print(wrd, in_word, out_word)
            in_out_stats[0][1] +=1
            
            
        if in_word!=tar_word and out_word==tar_word: ### making incorrect->correct
            #print(in_word, out_word)
            mct = sum(a!=b for a, b in zip(tar_word, in_word))
            if mct <= 4:
                in_out_stats[mct][0] +=1
            else:
                in_out_stats[5][0] +=1


        if in_word!=tar_word and out_word!=tar_word:  ### incorrect -> incorrect
            #print(in_word, out_word)
            mct = sum(a!=b for a, b in zip(tar_word, in_word))
            if mct <= 4:
                in_out_stats[mct][1] +=1
            else:
                in_out_stats[5][1] +=1
        

        
        mis_cnt = sum(a!=b for a, b in zip(tar_word, out_word))
        if mis_cnt > 0:
            err += 1
            mismatch_cnt[mis_cnt-1] += 1
            
        
 
        spellings.append(chars-1)
        


    err = err/num_words * 100
    
    #plt.matshow(cnt)

    return spellings, mismatch_cnt, err, in_out_stats



def CLSTM_evaluation(data, labels, model):
    

    test_data = copy.deepcopy(data)
    test_labels = copy.deepcopy(labels)
        
        
    num_words = len(test_data)
    #num_words = 500

    err = 0
    
    mismatch_cnt = np.zeros(20, dtype=int)
    
    spellings = []
    
    for wrd in range(num_words):
   
        input_tensor = test_data[wrd].unsqueeze(dim=1)
        target_tensor= test_labels[wrd]
        
        if device.type == "cuda":
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
        
        #initial hidden states
        h = torch.zeros(1+int(bi_dir), 1, hidden_dim_of_lstm1) ### if bi_dir=True, first dimension is 2 for bi-directional
        c = torch.zeros(1+int(bi_dir), 1, hidden_dim_of_lstm1)

        h = h.to(device)
        c = c.to(device)

        out, h, c = model(input_tensor, h, c)
        
        chars = out.argmax(dim=2)
        
        spellings.append(chars)
        
        out_word = [alphas[x] for x in chars]  #### convert labels to alphabets
        out_word = ''.join(out_word)
        
        tar_word = [alphas[x] for x in target_tensor]
        tar_word = ''.join(tar_word)

        if tar_word != out_word:
            err += 1
            
            mis_cnt = sum(a!=b for a, b in zip(tar_word, out_word))
            mismatch_cnt[mis_cnt-1] += 1
            

    err = err/num_words * 100

    return spellings, mismatch_cnt, err


def distortion_test(word_idx, num_of_distortions, testWordstestImages, data):
    
    idx = word_idx

    print('Original Word: ', testWordstestImages[idx][0])

    temp_label = []
    tl = torch.LongTensor([alphas.index(x) for x in list(testWordstestImages[idx][0])])
    temp_label.append(tl)

    temp_data = copy.deepcopy(data[idx])

    fig = plt.figure(figsize = (60,50))

    for j in range(len(temp_data)):
        plt.subplot(1, 20, j+1)
        im = plt.imshow(temp_data[j], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    
    print('\nDistorted Sequence')

    temp_data = copy.deepcopy(data[idx])

    ####################################### Introduce Distortion ##############################################
    for dtrt in range(num_of_distortions):
        
        at = random.randint(0, len(temp_data)-1)
        
        afine_tf = tf.AffineTransform(shear=0.8)

        temp_data[at] = torch.FloatTensor(tf.warp(temp_data[at], inverse_map=afine_tf))
        

    fig = plt.figure(figsize = (60,50))

    for j in range(len(temp_data)):
        plt.subplot(1, 20, j+1)
        im = plt.imshow(temp_data[j], cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.show()

    
    ################################### CHECK #####################################################################
    dat = copy.deepcopy(temp_data.reshape(1, len(temp_data), 28, 28))
    
    return dat, temp_label
    