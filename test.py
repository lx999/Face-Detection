# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:18:01 2019

@author: LI
"""
import numpy as np
from skimage import io, util, color, feature, transform
import os
from sklearn import svm
import pickle

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


######################################charger le modèle########################################
def load(filename):
    file = open(filename,'rb')
    obj = pickle.load(file)
    file.close()
    return obj

clf = load(r".\svc.pckl")

######################################aire confondue########################################
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

############generer detection.txt############
def sortie(txt_path, x_test):
    f = open(txt_path,'a') 
    for i in range(len(x_test)):
        print(i)
        result = fenetreglissante(util.img_as_float(color.rgb2gray(x_test[i])))
        coord = result[0]
        score = result[1]
    
        coord = nms(coord,score)[0]
        score = nms(coord,score)[1]
        for j in range(len(coord)):
            x = coord[j][0]
            y = coord[j][1]
            L = coord[j][2]
            H = coord[j][3]      
            f.writelines(['\n', "%03d"%(i+1) ,' ',str(y), ' ', str(x), ' ', str(H), ' ', str(L), ' ', str( float("{0:.2f}".format( score[j][0])) ) ]) 
    f.close()

txt_path = 'E:\GI05\SY32\SY32_projet\data\detection.txt'
xtst =  load_images_from_folder('E:\GI05\SY32\SY32_projet\data\\trainTest2')
sortie(txt_path, xtst)