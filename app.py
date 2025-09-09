from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)
net = None
CLASSES = ["background","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","chair","cow","diningtable",
           "dog","horse","motorbike","person","pottedplant",
           "sheep","sofa","train","tvmonitor"]

@app.route("/upload_model", methods=["POST"])
def upload_model():
    global net
    caffemodel = request.files["caffemodel"]
    prototxt = request.files["prototxt"]
    caffemodel.save("model.caffemodel")
    prototxt.save("deploy.prototxt")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "model.caffemodel")
    return jsonify({"status": "Model loaded successfully"})

@app.route("/detect", methods=["POST"])
def detect():
    global net
    if net is None:
        return jsonify({"error": "Model not loaded"}), 400

    data = request.json
    image_data = base64.b64decode(data["image"])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            results.append({
                "label": CLASSES[idx],
                "confidence": float(confidence),
                "box": [int(startX), int(startY), int(endX), int(endY)]
            })

    return jsonify({"detections": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
