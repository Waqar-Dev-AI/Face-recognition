from fastapi import FastAPI,File,UploadFile,Response
import cv2
import numpy as np
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import os 
from deepface import DeepFace
app=FastAPI()

model=YOLO("model.pt")
print(model.info())

@app.post('/detect_faces/')
async def detect_faces(file: UploadFile=File(...)):
    content =await file.read()
    np_img=np.frombuffer(content,np.uint8)
    img=cv2.imdecode(np_img,cv2.IMREAD_COLOR)

    results=model(img)
    faces=[]
    

    for result in results:
        for box in result.boxes.xyxy:
            x1,y1,x2,y2=map(int,box)
            faces.append(
                {"x1":x1,"y1":y1,"x2":x2,"y2":y2}
            )
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        pil_img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        img_io=BytesIO()
        pil_img.save(img_io,format="JPEG")
        img_io.seek(0)
        return Response (content=img_io.getvalue(),media_type="image/jpeg")



import uuid
import time
import base64  # For encoding images to send to UI
from fastapi.responses import JSONResponse
import json
from typing import List  # Import List from typing


from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # Import StaticFiles

app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

path = "Faces"  # Folder containing reference images
@app.post('/upload_reference_faces/')
async def upload_reference_faces(files: List[UploadFile] = File(...)):
    os.makedirs("Faces", exist_ok=True)
    for file in files:
        content = await file.read()
        file_path = os.path.join("Faces", file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
    return {"message": "Reference faces uploaded successfully"}
@app.post('/recognize_faces/')
async def detect_faces(file: UploadFile = File(...)):
    content = await file.read()
    np_img = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

    results = model(img)  # Run face detection model
    recognize_faces = []
    total_face = 0

    for i, result in enumerate(results):
        for j, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)

            # Ignore small/invalid faces
            if (x2 - x1) < 32 or (y2 - y1) < 32:
                print(f"Face {i}-{j}: Too small to process.")
                continue

            if y1 >= y2 or x1 >= x2 or x2 > img.shape[1] or y2 > img.shape[0]:
                print(f"Face {i}-{j}: Invalid bounding box ({x1}, {y1}, {x2}, {y2})")
                continue

            total_face += 1

            # Add margin to the face crop
            margin = 20
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)

            face = img[y1:y2, x1:x2]
            if face.size == 0:
                print(f"Face {i}-{j}: Empty crop detected.")
                continue

            # Save temp face image
            face_path = f"temp_face_{uuid.uuid4()}.jpg"
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            cv2.imwrite(face_path, face_rgb)

            # Validate if file is correctly saved
            if not os.path.exists(face_path) or os.stat(face_path).st_size == 0:
                print(f"Face {i}-{j}: Error - File not saved properly: {face_path}")
                os.remove(face_path)
                continue           

            matched_person = "No_match"
            confidence=100
            best_match = None
            lowest_distance = float('inf')
            best_confidence = 0
            # Compare detected face with reference faces
            for person_img in os.listdir(path):
                person_path = os.path.join(path, person_img)
                if not os.path.exists(person_path):
                    print(f"Reference image {person_path} not found.")
                    continue

                try:
                    # Use VGG-Face & Euclidean_L2
                    result = DeepFace.verify(face_path, person_path, model_name="ArcFace")
                                            #  , distance_metric="euclidean_l2",detector_backend="skip")
                    print(f"Face {i}-{j} vs {person_img}: {result}")

                    current_distance = result["distance"]
                    if result["verified"] and current_distance < lowest_distance:
                        lowest_distance = current_distance
                        best_match = person_img
                        matched_person = ''.join(filter(str.isalpha, person_img.split('.')[0]))
                        confidence = max(0, round((1 - (current_distance / result["threshold"])) * 100, 2))
                                    
                        
                except Exception as e:
                    print(f"Face {i}-{j} DeepFace error with {person_img}: {str(e)}")
                    continue

            recognize_faces.append({
                        "name": matched_person,  # Extract filename as name
                        "confidence": confidence,  # Convert to percentage
                        "box": [x1, y1, x2, y2]
                    })

               
            os.remove(face_path)  # Clean up temp file
             # **Draw Bounding Box and Label**
            color = (0, 255, 0) if matched_person != "No_match" else (0, 0, 255)  # Green for match, Red for no match
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, matched_person, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3, cv2.LINE_AA)           
    # # **Convert Image to Response**
    # _, img_encoded = cv2.imencode('.jpg', img)
    # return Response(content=img_encoded.tobytes(), media_type="image/jpeg")
      # **Convert Image to Response**
    _, img_encoded = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")  # Convert to Base64 string
    image_path = "output.jpg"
    with open(image_path, "wb") as f:
        f.write(img_encoded.tobytes())

    # Return JSON with a downloadable image link
    return JSONResponse(
        content={
            "recognized_faces": recognize_faces,  # Your face recognition results
            "image_url": f"http://127.0.0.1:8000/static/output.jpg"  # Public URL for 
            
        }
    )


















#vgg-face
# @app.post('/recognize_faces/')
# async def detect_faces(file: UploadFile = File(...)):
#     content = await file.read()
#     np_img = np.frombuffer(content, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     if img is None:
#         return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

#     results = model(img)  # Run face detection model
#     recognize_faces = []
#     total_face = 0

#     for i, result in enumerate(results):
#         for j, box in enumerate(result.boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box)

#             # Ignore small/invalid faces
#             if (x2 - x1) < 32 or (y2 - y1) < 32:
#                 print(f"Face {i}-{j}: Too small to process.")
#                 continue

#             if y1 >= y2 or x1 >= x2 or x2 > img.shape[1] or y2 > img.shape[0]:
#                 print(f"Face {i}-{j}: Invalid bounding box ({x1}, {y1}, {x2}, {y2})")
#                 continue

#             total_face += 1

#             # Add margin to the face crop
#             margin = 20
#             x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
#             x2, y2 = min(img.shape[1], x2 + margin), min(img.shape[0], y2 + margin)

#             face = img[y1:y2, x1:x2]
#             if face.size == 0:
#                 print(f"Face {i}-{j}: Empty crop detected.")
#                 continue

#             # Save temp face image
#             face_path = f"temp_face_{uuid.uuid4()}.jpg"
#             face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             cv2.imwrite(face_path, face_rgb)

#             # Validate if file is correctly saved
#             if not os.path.exists(face_path) or os.stat(face_path).st_size == 0:
#                 print(f"Face {i}-{j}: Error - File not saved properly: {face_path}")
#                 os.remove(face_path)
#                 continue           

#             matched_person = "No_match"

#             # Compare detected face with reference faces
#             for person_img in os.listdir(path):
#                 person_path = os.path.join(path, person_img)
#                 if not os.path.exists(person_path):
#                     print(f"Reference image {person_path} not found.")
#                     continue

#                 try:
#                     # Use VGG-Face & Euclidean_L2
#                     result = DeepFace.verify(face_path, person_path, model_name="VGG-Face", distance_metric="euclidean_l2")
#                     print(f"Face {i}-{j} vs {person_img}: {result}")

#                     if result["verified"]:
#                         matched_person = person_img.split('.')[0]
#                         break
#                 except Exception as e:
#                     print(f"Face {i}-{j} DeepFace error with {person_img}: {str(e)}")
#                     continue

#             recognize_faces.append(matched_person)
#             os.remove(face_path)  # Clean up temp file
#              # **Draw Bounding Box and Label**
#             color = (0, 255, 0) if matched_person != "No_match" else (0, 0, 255)  # Green for match, Red for no match
#             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(img, matched_person, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # **Convert Image to Response**
#     _, img_encoded = cv2.imencode('.jpg', img)
#     return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

    # return {
    #     "total_face_detected": total_face,
    #     "recognized_faces": recognize_faces,
    # }



#Balanced (speed + accuracy)	FaceNet, SFace
# @app.post('/recognize_faces/')
# async def detect_faces(file: UploadFile=File(...)):
#     content =await file.read()
#     np_img=np.frombuffer(content,np.uint8)
#     img=cv2.imdecode(np_img,cv2.IMREAD_COLOR)

#     results=model(img)
#     recognize_faces=[]
    

#     for result in results:
#         for box in result.boxes.xyxy:
#             x1,y1,x2,y2=map(int,box)
#             face=img[y1:y2,x1:x2]
#             face_path="temp_face.jpg"
#             cv2.imwrite(face_path,face)
#             matched_person="Unknown"
#             for person_img in os.listdir(path):
#                 person_path=os.path.join(path,person_img)
#                 try:
#                     result=DeepFace.verify(face_path,person_path,model_name="Facenet")
#                     if result["verified"]:
#                         matched_person=person_img.split('.')[0]
#                         break
#                 except:
#                     continue
#             recognize_faces.append(matched_person)
            

#             return{"total face": len(recognize_faces),
#                    "recognize face":recognize_faces
                   
#                    }
                    
                        
# @app.post('/recognize_faces/')
# async def detect_faces(file: UploadFile = File(...)):
#     content = await file.read()
#     np_img = np.frombuffer(content, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     results = model(img)
#     recognize_faces = []
#     total_face=0
#     for i, result in enumerate(results):
#         for j, box in enumerate(result.boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box)

#             # if (x2 - x1) < 32 or (y2 - y1) < 32:
#             #     print(f"Face {i}-{j}: Too small to process.")
#             #     continue
#             total_face+=1
#             face = img[y1:y2, x1:x2]
#             face_path = f"temp_face_{i}_{j}.jpg"
#             face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
#             cv2.imwrite(face_path, face_rgb)

#             if not os.path.exists(face_path):
#                 print(f"Face {i}-{j}: Save failed for {face_path}.")
#                 continue

#             test_img = cv2.imread(face_path)
#             if test_img is None or test_img.size == 0:
#                 print(f"Face {i}-{j}: Saved image {face_path} is corrupt.")
#                 continue

#             matched_person = "Unknown"
#             for person_img in os.listdir(path):
#                 person_path = os.path.join(path, person_img)
#                 if not os.path.exists(person_path):
#                     print(f"Reference image {person_path} not found.")
#                     continue

#                 try:
#                     result = DeepFace.verify(face_path, person_path, model_name="Facenet")
#                     print(f"Face {i}-{j} vs {person_img}: {result}")
#                     if result["verified"]:
#                         matched_person = person_img.split('.')[0]
#                         break
#                 except Exception as e:
#                     print(f"Face {i}-{j} DeepFace error with {person_img}: {str(e)}")
#                     continue

#             if matched_person != "Unknown":
#                 recognize_faces.append(matched_person)
#             os.remove(face_path)  # Clean up temp file

#     return {
#         "total face": total_face,
#         "recognize face": recognize_faces
#     }
