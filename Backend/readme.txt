yolov8 se face detect kia ha 
phir deepface library se ki ha recognitaion kelye
DeepFace ek powerful AI-based face recognition library hai jo pre-trained deep learning models ka use karti hai. 
Isme face verification, detection, recognition, emotion analysis, age & gender detection sab kuch available hai.

from deepface import DeepFace
result = DeepFace.verify("image1.jpg", "image2.jpg")
print(result)

{'verified': True, 'distance': 0.25, 'threshold': 0.4}


custom actions 
	result = DeepFace.analyze("image.jpg", actions=['emotion'])
	print(result)
	{'dominant_emotion': 'happy'}

custom model
	result = DeepFace.verify("img1.jpg", "img2.jpg", model_name="ArcFace")


yeah deepface apna build in facedetection use krta ha yha mn yolo use krrha hu to iski detection ko skip krna hoga
            result = DeepFace.verify(face_path, person_path, model_name="ArcFace", distance_metric="euclidean_l2",detector_backend="skip")

better accuracy kelye preprocessing krskty ha ,size ,grayscale , alignment etc
