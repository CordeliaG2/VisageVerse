import cv2
import os
import numpy as np
import tkinter as tk
import time
from threading import Thread
from datetime import datetime, timedelta

TIEMPO_ENTRE_DETECCIONES = 60
ultima_deteccion = None
persona_detectada = ""
archivo_creado = False
estado_personas = {}
#etiqueta_estado = None 

def mostrar_informacion(persona, estado):
    if persona is not None:
        evento = "Entrada" if estado else "Salida"
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        informacion = f"Persona: {persona}\nFecha y hora: {fecha_hora}\nEstado: {evento}"
        print(informacion)  # Puedes reemplazar esto con la lógica para mostrar la información en una ventana emergente


def guardar_en_archivo(persona, estado):
    if persona is not None:
        evento = "Entrada" if estado else "Salida"
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("detecciones.txt", "a") as file:
            file.write(f"Persona detectada: {persona}, Fecha y hora: {fecha_hora},{evento}\n")

def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))


def solicitar_nombre_apellido():
    ventana_nombre_apellido = tk.Toplevel(ventana)
    ventana_nombre_apellido.title("Ingresar nombre y apellido")

    etiqueta_nombre = tk.Label(ventana_nombre_apellido, text="Nombre:")
    etiqueta_nombre.pack()
    entry_nombre = tk.Entry(ventana_nombre_apellido)
    entry_nombre.pack()

    etiqueta_apellido = tk.Label(ventana_nombre_apellido, text="Apellido:")
    etiqueta_apellido.pack()
    entry_apellido = tk.Entry(ventana_nombre_apellido)
    entry_apellido.pack()

    def guardar_nombre_apellido():
        nombre = entry_nombre.get()
        apellido = entry_apellido.get()
        global personName
        personName = f"{nombre}_{apellido}"
        ventana_nombre_apellido.destroy()
        case2()

    boton_guardar = tk.Button(ventana_nombre_apellido, text="Guardar", command=guardar_nombre_apellido)
    boton_guardar.pack()


# Función para cargar el modelo en un hilo separado
def cargar_modelo():
    global face_recognizer, etiqueta_estado
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    if os.path.exists('models/modeloLBPHFace.xml'):
        face_recognizer.read('models/modeloLBPHFace.xml')
        print("Modelo cargado exitosamente")
        etiqueta_estado.config(text="Modelo cargado exitosamente")
        boton_reconocimiento.config(state=tk.NORMAL)  # Activamos el botón "Modo Reconocimiento"
        #ventana.after(0, abrir_modo_reconocimiento)  # Llamamos a abrir_modo_reconocimiento después de cargar el modelo

def abrir_modo_reconocimiento():
    case1()

def abrir_modo_entrenamiento():
    solicitar_nombre_apellido()

def case1():
    """
    print("Iniciando detección de personas...")
    dataPath = 'Data'
    imagePaths = os.listdir(dataPath)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.read('models/modeloLBPHFace.xml')
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: No se puede acceder a la cámara.")
        return

    faceClassif = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    persona = None 
    while True:
        ret,frame = cap.read()
        if ret == False:
            print("Error: No se pudo capturar el fotograma.")
            break
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            if result[1] < 70:
                persona = imagePaths[result[0]]
                guardar_en_archivo(persona)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame,'{}'.format(persona),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        cv2.imshow('frame',frame)

        k = cv2.waitKey(1)
        if k == 27:
            break
    """
    global persona_detectada, archivo_creado, ultima_deteccion, estado_personas
    dataPath = 'Data'
    imagePaths = os.listdir(dataPath)
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.read('models/modeloLBPHFace.xml')
    
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    faceClassif = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    while True:
        ret,frame = cap.read()
        if ret == False: break
        frame = cv2.flip(frame,1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray,1.3,5)
        persona = None 
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)

            cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
            
            if result[1] < 70:
                index = result[0]  # Obtener el índice de la predicción
                if 0 <= index < len(imagePaths):
                    persona = imagePaths[index]
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                    if ultima_deteccion is None or datetime.now() - ultima_deteccion >= timedelta(seconds=TIEMPO_ENTRE_DETECCIONES):
                        if persona in estado_personas:
                            estado = not estado_personas[persona]  # Cambiar el estado de entrada a salida y viceversa
                        else:
                            estado = True
                        print(f"Persona detectada: {persona}")
                        guardar_en_archivo(persona, estado)
                        ultima_deteccion = datetime.now()
                        estado_personas[persona] = estado
                        mostrar_informacion(persona, estado)
                else:
                    print("Error: Índice fuera de rango en imagePaths.")
            else:
                persona = None
                cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('frame',frame)

        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def case2():
    global personName  # Asegúrate de usar la variable global personName
    #personName = 'primita'
    dataPath = 'Data'
    personPath = dataPath + '/' + personName

    if personName is None:
        solicitar_nombre_apellido()
        return 

    if not os.path.exists(personPath):
        print('Carpeta creada: ',personPath)
        os.makedirs(personPath)
    
    face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_defaultC.xml")

    """cap = cv2.VideoCapture('Video.mp4')"""
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    count = 0

    while True:
        _, img = cap.read()
        img = cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        auxFrame=img.copy()
        faces = face_cascade.detectMultiScale(gray,
        scaleFactor=1.23,
        minNeighbors=5,
        minSize=(40,40),
        maxSize=(200,200))
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)
            rostro = auxFrame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
            count = count + 1
        cv2.imshow('img', img)
        
        k = cv2.waitKey(1)
        if k == 27 or count>=100:
            break

    dataPath = 'Data'
    peopleList = os.listdir(dataPath)
    print('Lista de personas: ', peopleList)

    labels = []
    facesData = []
    label = 0

    for nameDir in peopleList:
        personPath = dataPath + '/' + nameDir
        print('Leyendo las imágenes')

        for fileName in os.listdir(personPath):
            print('Rostros: ', nameDir + '/' + fileName)
            labels.append(label)
            facesData.append(cv2.imread(personPath+'/'+fileName,0))
        label = label + 1

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))

    face_recognizer.write('models/modeloLBPHFace.xml')
    print("Modelo almacenado...")

def case3():
    faceClassif = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    while True:
        _, image = cap.read()
        image = cv2.flip(image,1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceClassif.detectMultiScale(gray,1.1,4)
        cv2.putText(image,'Se encontro un error: Modo seguro',(10,20), 2, 0.5,(128,0,255),1,cv2.LINE_AA)
        
        for(x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w, y+h),(255,0,0),2)
        cv2.imshow('image', image)
        
        k = cv2.waitKey(1)
        if k == 27:
            break

def main():
    global ventana, etiqueta_estado, boton_reconocimiento

    hilo_carga_modelo = Thread(target=cargar_modelo)
    hilo_carga_modelo.start()
    
    ventana = tk.Tk()
    ventana.title("Interfaz de Usuario")

    etiqueta = tk.Label(ventana, text="Selecciona una opción:")
    etiqueta.pack()

    boton_reconocimiento = tk.Button(ventana, text="Modo Reconocimiento", command=abrir_modo_reconocimiento, state=tk.DISABLED)  # Inicialmente desactivado
    boton_reconocimiento.pack()

    boton_entrenamiento = tk.Button(ventana, text="Modo Entrenamiento", command=abrir_modo_entrenamiento)
    boton_entrenamiento.pack()

    boton_opcion3 = tk.Button(ventana, text="Opción 3", command=case3)
    boton_opcion3.pack()

    etiqueta_estado = tk.Label(ventana, text="Estado del modelo: Cargando...")
    etiqueta_estado.pack()

    ventana.mainloop()

if __name__ == "__main__":
    main()
