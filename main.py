import cv2
import os
import imutils

personName = 'NombreDeLaPersona'
#Aca va las imagenes para el dataset
dataPath = 'D:/prueba1/Data' 
#Esto para guardar una carpeta con cada nombre de las personas
personPath = dataPath + '/' + personName

#Controlo que no exista la carpeta, si no existe la creo
if not os.path.exists(personPath):
	print('Carpeta creada: ',personPath)
	os.makedirs(personPath)

#El primero es para carpturar en tiempo real, el segundo es para darle un video pre grabado
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('D:/prueba1/VideoDePrueba/nombre.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#Contador para saber cuantas fotos tomamos
count = 0

while True:

	ret, frame = cap.read()
	if ret == False: break
	frame =  imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	auxFrame = frame.copy()

	faces = faceClassif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = auxFrame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		#Para mostrar las fotos si es necesario
		#cv2.imshow("image", rostro)
		#cv2.waitKey(30)
		personPath2 = personPath + '/rostro_' + str(count)+'.bmp'
		#Guarda la foto
		cv2.imwrite(str(personPath2),rostro)
		count = count + 1
	cv2.imshow('frame',frame)

	k =  cv2.waitKey(1)
	#Si apreto escape, que su numero es 27, o llego a 200 fotos, paro.
	if k == 27 or count >= 200:
		break

cap.release()
cv2.destroyAllWindows()