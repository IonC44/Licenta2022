# -*- coding: utf-8 -*-
"""
Creem o clasa ce va avea diferite metode de preprocesare -> In acest
caz vom decupa imaginiile astfel incat sa obtinem o dimensiune egala
"""

#importam pachetul opencv
import cv2

class preprocesor:
    
    def __init__(self,width,height, inter = cv2.INTER_AREA):
        #salvam dimensiunea imaginii, si metoda de interpolare(indicele ei)
        
        self.width = width
        self.height = height
        self.inter = inter
    
    def redimensionare(self,image):
        #redimensionam imaginea folosind metoda resize
        return cv2.resize(image,(self.width,self.height),
                          interpolation = self.inter)
    