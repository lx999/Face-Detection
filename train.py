# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:23:38 2019

@author: LI
"""

import numpy as np
from skimage import io, util, color, feature, transform
import os
from random import randint
from sklearn import svm

######################################charger de données######################################
def load_images_from_folder(folder):
    images = []
    files =os.listdir(folder)
    files.sort()
    for filename in files:
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def read_images_to_resize_gray_and_float_and_hog_from_folder(folder):
    images = []
    files =os.listdir(folder)
    files.sort()
    for filename in files:
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            img = feature.hog(util.img_as_float(transform.resize(color.rgb2gray(img), [48,32])),block_norm='L2-Hys')
            images.append(img)
    return images

x_train = load_images_from_folder('E:\GI05\SY32\SY32_projet\data\origin')
x_label = np.loadtxt(r"E:\GI05\SY32\SY32_projet\data\label.txt")

###################################### positives et négatives##################################
pos_path = '/Users/yann/Desktop/UTC/GI05/SY32_projet/data/pos/'
neg_path = '/Users/yann/Desktop/UTC/GI05/SY32_projet/data/neg/'
######################################extract n positive miages with format exact#####################################
i = 0
while i < x_label.shape[0]:
    index = int(x_label[i,0]-1)
    cropped = x_train[index][int(x_label[i][1]):int(x_label[i][1]+x_label[i][3]), int(x_label[i][2]):int(x_label[i][2]+x_label[i][4])]
    s=pos_path + '%d'%(i)+ '.jpg'
    io.imsave(s, cropped) 
    i += 1

######################################extract 10000 negative images#####################################
def aireJoinUnion(rc1x, rc1y, rc1L, rc1H, rc2x, rc2y, rc2L, rc2H):
    p1x = max(rc1x, rc2x)
    p1y = max(rc1y, rc2y)
    p2x = min(rc1x + rc1L, rc2x + rc2L)
    p2y = min(rc1y + rc1H, rc2y + rc2H)
    AJoin = 0
    if p2x >= p1x and p2y >= p1y:
        AJoin = (p2x - p1x)*(p2y - p1y)
    else:
        return 0
    A1 = rc1L*rc1H
    A2 = rc1L*rc2H
    if AJoin/A1 > 0.8 or AJoin/A2 > 0.8:
        return 1
    AUnion = A1 + A2 - AJoin
    return AJoin/AUnion

# test si un image est un HeadPortrait ou pas
def ispHeadPortrait(im,L, H):
    isHeadPortrait = False
    l = im.shape[1]
    h = im.shape[0]
    s_im = l * h
    s_extract = L * H
    if (s_extract/s_im ) > 0.4:
        isHeadPortrait = True
    return isHeadPortrait

def max_Label(lista , LOH):
    max_label = 0;
    for i in lista:
        if max_label < i[LOH]:
            max_label = i[LOH]
    return max_label

def avg_Label(lista , LOH):
    avg = 0
    for i in lista:
        avg = avg + i[LOH]
    return int(avg/(len(lista)))
        

def extractneg(im, labels, nb, indexIm):
    
    i = 0
    taille = im.shape
    L = max_Label(labels,4)
    H = max_Label(labels,3)
    if ispHeadPortrait(im, L, H):
        L = int(L/2)
        H = int(H/2)
        while i< nb:
            xrand = randint(0,taille[1]-L)
            yrand = randint(0,taille[0]-H)
            cropped = im[yrand:int(yrand+H),xrand:int(xrand+L)]
            s = neg_path + '%d'%(indexIm)+ '.jpg'
            io.imsave(s, cropped)
            i += 1
            indexIm += 1
            print(indexIm)
    else:     
        while i < nb:
            isVisage = False
            xrand = randint(0,taille[1]-L)
            yrand = randint(0,taille[0]-H)
            for label in labels:
                ratioAireJoinUnion = aireJoinUnion(xrand, yrand, label[4], label[3], label[2], label[1], label[4], label[3])
                if ratioAireJoinUnion >= 0.5:
                    isVisage = True
                if isVisage == True:
                    break
            if isVisage == False :
                l_extract = avg_Label(labels,4)
                h_extract = avg_Label(labels,3)
                cropped = im[yrand:int(yrand+h_extract),xrand:int(xrand+l_extract)]
                s = neg_path + '%d'%(indexIm)+ '.jpg'
                io.imsave(s, cropped)
                i += 1
                indexIm += 1
                print(indexIm)
    

n_train = 1000
indexImage = 1
indexLabel = 0
numImage = 0
for image in x_train[0:n_train]:
    currLabel = list()
    while x_label[indexLabel][0] == indexImage:
        currLabel.append(x_label[indexLabel])
        indexLabel += 1
        if indexLabel >= x_label.shape[0]:
            break
    test = currLabel
    extractneg(image, currLabel, 10, numImage)
    numImage += 10
    indexImage += 1
    
    

######################################aprentissage#####################################
pos_path="E:\GI05\SY32\SY32_projet\data\pos"
neg_path="E:\GI05\SY32\SY32_projet\data\\neg"

trainPos = read_images_to_resize_gray_and_float_and_hog_from_folder(pos_path)
trainNeg = read_images_to_resize_gray_and_float_and_hog_from_folder(neg_path)
dataTrain = trainPos + trainNeg
dataTrain = np.ravel(dataTrain).reshape(len(trainNeg) + len(trainPos),648) # len(ravel(dataTrain))/(len(trainNeg) + len(trainPos))
yTrain = np.concatenate((np.ones(len(trainPos)),-np.ones(len(trainNeg))))

def CrossValidation(xtrain, ytrain, Nvc):
    clf = svm.SVC(kernel='linear')
    r = np.zeros(Nvc)
    for i in range(Nvc):
        print("boucle : %d"%(i))
        mask = np.zeros(xtrain.shape[0], dtype = bool)
        mask[np.arange(i, mask.size, Nvc)] = True
        clf.fit(xtrain[~mask], ytrain[~mask])
        r[i] = np.mean(clf.predict(xtrain[mask]) != ytrain[mask])
    print("Taux d'erreur : %d"%(np.mean(r)))
    return clf

clf = CrossValidation(dataTrain, yTrain , 5)
######################################fenetreglissante#####################################
def fenetreglissante(im):
    listxy = []
    listscore = []
    scale = [0.5,0.4,0.3,0.2,0.1]
#min fenetre
    resL =32
    resH =48
    pas = 8
    for k in scale:
        imscaled = transform.rescale(im, k, mode='reflect',multichannel=False, anti_aliasing=True, anti_aliasing_sigma=None)
        longueur = np.shape(imscaled)[1]
        hauteur = np.shape(imscaled)[0]
        for y in range(0, hauteur-resH,pas):
            for x in range(0, longueur-resL, pas):
                if(x+resL < longueur and y+resH < hauteur):
                    carre = imscaled[y:y+resH, x:x+resL]
                    carre = feature.hog(carre,block_norm='L2-Hys')
                    carrepredict = clf.predict(np.ravel(carre).reshape(1,648))
                    score = clf.decision_function(np.ravel(carre).reshape(1,648))
                    if(carrepredict[0] == 1 and  score > 0.45 ):
                        listxy.append(((np.array([x, y, resL,resH])* (1/k)).astype(int)))  
                        listscore.append(np.array(score)) 
    return listxy , listscore

######################################suppression superposition############################
def nms(coords, scores):
    i_base = 0
    while i_base < len(coords):
        i_new = 0
        while i_new < len(coords) and i_base < len(coords):       
            ratioAireJoinUnion = aireJoinUnion(coords[i_base][0],coords[i_base][1],coords[i_base][2],coords[i_base][3],coords[i_new][0],coords[i_new][1],coords[i_new][2],coords[i_new][3])
            if ratioAireJoinUnion > 0.5:
                if scores[i_new][0] > scores[i_base][0]:
                    del coords[i_base]
                    del scores[i_base]
                    i_base -= 1
                    break
                elif scores[i_new][0] < scores[i_base][0]:
                    del coords[i_new]
                    del scores[i_new]
                    i_new -= 1
                    
            i_new += 1
        i_base += 1
    return coords, scores


######################################Faux Positive#########################################
def falsePosToNeg(clf, trainFiles, label, path):
    print("Starting generation of false positive")
    indexImage = 1
    for i in range(len(trainFiles)):  
        print(i)              
        testedImg = trainFiles[i]
        testedImg = util.img_as_float(color.rgb2gray(testedImg))
        result = fenetreglissante(testedImg)
        coord = result[0]
        score = result[1]
        for lab in label:
            
            if lab[0] == (i+1):
#x, y, resL,resH
#label: index, y,x,resh,resl
                coord.append(np.array([lab[2],lab[1],lab[4],lab[3]]))                
                score.append(np.array([99999]))

        coord = nms(coord,score)[0]
        score = nms(coord,score)[1]
        for j in range(len(score)):
            if score[j][0]<9999:
#                H,L
                s = path+ "%03d"%indexImage + '.jpg'
                io.imsave(s, trainFiles[i][coord[j][1]:(coord[j][1]+coord[j][3]),coord[j][0]:(coord[j][0]+coord[j][2])])
                indexImage = indexImage + 1
    print("False Positive added to neg")

newNegPath = 'E:\GI05\SY32\SY32_projet\data\\newNeg\\'
falsePosToNeg(clf, x_train, x_label,newNegPath)