# Khai báo các thư viện
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
#Chương trình con biểu diễn thuật toán
def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))


			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Chỉ chạy thuật toán khi phát hiện được khuôn mặt
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)


print("[Processing] Tải mô hình phát hiện khuôn mặt...")

prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector","res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[Processing] Hoàn tất...")

print("[Processing] Tải mô hình phân loại khuôn mặt...")

baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
Classifier_Net = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

Classifier_Net.load_weights("./face_classifier/face_classifier.h5")

print("[Processing] Hoàn tất...")

print("[Processing] Khởi động máy ảnh...")
vs = VideoStream(src=0).start()
#Đợi 3s để khởi động webcam
time.sleep(3.0)

#Chạy chương trình
while True:
	webcam = vs.read()
	fps = FPS().start()
	webcam = imutils.resize(webcam, width=400)
	(locs, preds) = detect_and_predict_mask(webcam, faceNet, Classifier_Net)


	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(webcam, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(webcam, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Webcam", webcam)
	key = cv2.waitKey(1) & 0xFF
	fps.update()
	if key == ord("q"):
		break
fps.stop()
print("[Processing] Thời gian chạy: {:.2f}".format(fps.elapsed()))
print("[Processing] FPS xấp xỉ: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()
