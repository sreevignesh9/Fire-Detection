from flask import Flask,request
import pandas as pd
import numpy as np
import base64
import cv2
import joblib
import json
import skimage
from skimage.transform import resize
from keras.models import model_from_json

app=Flask(__name__)

@app.route('/')
def test():
	return "Server is Up"

@app.route("/checkingfire",methods=["GET", "POST"])
def checkingfire():
	if request.method=="POST" or request.method == "GET":
		data=request.get_json()
		uri = data["image"]
		# print(uri)
		encoded_data = uri.split(',')[1]
		nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
		img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
		# print(img)
		model = joblib.load("my_random_forest.joblib")
		json_file = open('future_extractor.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		future_extractor = model_from_json(loaded_model_json)
		# load weights into new model
		future_extractor.load_weights("future_extractor.h5")
		print("Loaded model from disk")
		x = []
		img = skimage.transform.resize(img,(128,128,3), mode = "constant",anti_aliasing=True)
		img_arr = np.asarray(img)
		x.append(img_arr)
		x = np.asarray(x)
		X_test_feature = future_extractor.predict(x)
		prediction_RF = model.predict(X_test_feature)
		if prediction_RF[0] == 0:
			res = "NoFire"
		else:
			res = "Fire"
		response = app.response_class(
        response=json.dumps(res),
        status=200,
        mimetype='application/json'
    )
	return response

if __name__=='__main__':
	app.run(host="0.0.0.0", port=7778, debug = True)