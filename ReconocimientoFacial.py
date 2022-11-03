import cv2
import os
import imutils

dataPath = 'D:/prueba1/Data'
imagePaths = os.listdir(dataPath)
print('imagePaths', imagePaths)

#face_recognizer = cv2.face.LBPHFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#Leyendo el modelo
#face_recognizer.read('D:/prueba1/modeloEigenFace3.yml')
#face_recognizer.read('D:/prueba1/modeloFisherFace1.yml')
face_recognizer.read('D:/prueba1/modeloLBPHFace.xml')

#cap = cv2.VideoCapture('D:/prueba1/VideoDePrueba/Matias La Pioggia.mp4')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#comando para leer imagen
#cap = cv2.imread("D:/prueba1/VideoDePrueba/mauroimg.jpg", 1)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
access = False
while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = gray.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation = cv2.INTER_CUBIC)
		#cv2.imshow("image", rostro)
		#cv2.waitKey(60)
        
		result = face_recognizer.predict(rostro)

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

		# LBPHFace
		if result[1] < 70: # Si entro aca, es que el sistema encontro una cara que tenemos en el dataset
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			access = True # Eso quiere decir que hay que darle acceso
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)

	cv2.imshow('frame',frame)
	k = cv2.waitKey(1)
	#SIN ACCESO
	if k == 27:
		break

	#CON ACCESO
"""
	if k == 27 or access:
		if access: #Si el acceso fue consedido, cerramos la ventana y le damos acceso
			print('BIENVENIDO '+imagePaths[result[0]])
		else:
			print('Ventana cerrada')
		break
"""
cap.release()
cv2.destroyAllWindows()