import cv2


"""faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread('paunegro.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = faceClassif.detectMultiScale(gray,
  scaleFactor=1.1,
  minNeighbors=5,
  minSize=(30,30),
  maxSize=(200,200))
for (x,y,w,h) in faces:
  #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
  face_image = image[y:y+h,x:x+w]
  #cv2.imwrite("face.jpg",face_image)
  resized = cv2.resize(face_image,(48,48))
  cv2.imwrite("resized_face.jpg", resized)



cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

class FaceRecognition():
  
    def __init__(self):
       self.faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


    def get_faces(self,img_path):
      image = cv2.imread(img_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = self.faceClassif.detectMultiScale(gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        maxSize=(200,200))
      return image,faces
    
    def detect_faces(self,img_path):
      image, faces = self.get_faces(img_path)

      faces_list = []
      for (x,y,w,h) in faces:
        face_image = image[y:y+h,x:x+w]
        faces_list.append(cv2.resize(face_image,(48,48)))

      return faces_list


face_rec = FaceRecognition()

for faces in face_rec.detect_faces('orla16-17b.jpg'):
  cv2.imshow('image',faces)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

