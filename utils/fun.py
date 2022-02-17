#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:47:33 2021

@author: ngaggion
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

## TO VECTOR FORM AND BACK

def genVector(rl, ll, h, rc, lc):
    rl = rl.reshape(-1) 
    ll = ll.reshape(-1) 
    h = h.reshape(-1) 
    rc = rc.reshape(-1) 
    lc = lc.reshape(-1) 
    
    return np.concatenate([rl, ll, h, rc, lc])
    
def reverseVector(vector):
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    RCLAV = 23
    #LCLAV = 23
    
    p1 = RLUNG*2
    p2 = p1 + LLUNG*2
    p3 = p2 + HEART*2
    p4 = p3 + RCLAV*2
    
    rl = vector[:p1].reshape(-1,2)
    ll = vector[p1:p2].reshape(-1,2)
    h = vector[p2:p3].reshape(-1,2)
    rc = vector[p3:p4].reshape(-1,2)
    lc = vector[p4:].reshape(-1,2)
    
    return rl, ll, h, rc, lc

## DRAWING ON AXES

#important vertex
RL = [0, 21, 29]
LL = [0, 21, 27, 40, 44]
H = [0, 6, 12, 18]

def draw_organ(ax, array, color = 'b', bigger = None):
    N = array.shape[0]
    
    for i in range(0, N):
        x, y = array[i,:]
        
        if bigger is not None:
            if i in bigger:
                circ = plt.Circle((x, y), radius=9, color=color, fill = True)
                ax.add_patch(circ)
                circ = plt.Circle((x, y), radius=3, color='white', fill = True)
            else:
                circ = plt.Circle((x, y), radius=3, color=color, fill = True)
        else:
            circ = plt.Circle((x, y), radius=3, color=color, fill = True)
            
        ax.add_patch(circ)
    return

def draw_lines(ax, array, color = 'b'):
    N = array.shape[0]
    
    for i in range(0, N):
        x1, y1 = array[i-1,:]
        x2, y2 = array[i,:]
        
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
    return

def drawOrgans(ax, vector, title = '', img =  None):
    
    vector = vector.reshape(-1, 1)
    right_lung, left_lung, heart, right_clavicle, left_clavicle = reverseVector(vector)
    
    if img is not None:
        plt.imshow(img, cmap='gray')
    
    #ax.set_ylim(1024, 0)
    #ax.set_xlim(0, 1024)
    #fig.gca().set_aspect('equal', adjustable='box')
    
    draw_lines(ax, right_lung, 'r')
    draw_lines(ax, left_lung, 'g')
    draw_lines(ax, heart, 'y')
    draw_lines(ax, right_clavicle, 'b')
    draw_lines(ax, left_clavicle, 'darkorange')
    
    draw_organ(ax, right_lung, 'r', RL)
    draw_organ(ax, left_lung, 'g', LL)
    draw_organ(ax, heart, 'y', H)
    draw_organ(ax, right_clavicle, 'b')
    draw_organ(ax, left_clavicle, 'darkorange')

    ax.set_title(title)
    
    return

## imagen binaria

def drawBinary(img, organ):
    contorno = organ.reshape(-1, 1, 2)

    contorno = contorno.astype('int')
    
    img = cv2.drawContours(img, [contorno], -1, 255, -1)
    
    return img
