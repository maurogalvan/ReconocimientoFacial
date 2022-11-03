import cv2
import os
import numpy as np

dataPath = 'D:/prueba1/Data'
peopleList = os.listdir(dataPath)

labels = []
facesData = []
label  = 0

for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    #print ('Leyendo las imagenes')
    
    for fileName in os.listdir(personPath):
        #print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath+'/'+fileName,0))
        image = cv2.imread(personPath+'/'+fileName,0)
        #cv2.imshow('image', image)
        #cv2.waitKey(10)
    label = label + 1
#print('labels= ', labels)
#print('numero de etiquetas en 0: ', np.count_nonzero(np.array(labels)==0))
#print('numero de etiquetas en 1: ', np.count_nonzero(np.array(labels)==1))

#Entrenamiento

# Para que funcione hay que instalar -> pip install opencv-contrib-python
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('D:/prueba1/modeloEigenFace1.yml')
#face_recognizer.write('D:/prueba1/modeloFisherFace1.xml')
face_recognizer.write('D:/prueba1/modeloLBPHFace.xml')
print('*** Modelo almacenado *** ')