from tensorflow import keras
import os
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import model_from_json
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class FireDetection:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test


	def train_model(self, batch_size, epochs):

		feature_extractor = models.Sequential()

		feature_extractor.add(layers.Conv2D(64,(3,3),activation="relu",input_shape=(128,128,3)))
		feature_extractor.add(layers.MaxPooling2D((2,2)))
		feature_extractor.add(BatchNormalization())
		feature_extractor.add(Dropout(0.5))

		feature_extractor.add(layers.Conv2D(64,(3,3),activation="relu"))
		feature_extractor.add(layers.MaxPooling2D((2,2)))

		feature_extractor.add(layers.Conv2D(64,(3,3),activation="relu"))
		feature_extractor.add(layers.MaxPooling2D((2,2)))

		feature_extractor.add(layers.Flatten())

		x = feature_extractor.output
		x = (layers.Dense(128,activation="relu"))(x)
		x = (BatchNormalization())(x)
		x = (Dropout(0.5))(x)

		pred_layer = (layers.Dense(2,activation="softmax"))(x)

		cnn_model = Model(inputs=future_extractor.input, outputs=pred_layer)

		cnn_model.compile(optimizer = "adam" , loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

		history =  cnn_model.fit(self.X_train, self.y_train, batch_size=batch_size,epochs=epochs,verbose=1,validation_data =(self.X_test,self.y_test))


		self.save_feature_extractor(feature_extractor)

		return feature_extractor

	def load_feature_extractor(self):
		json_file = open('future_extractor.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		feature_extractor = model_from_json(loaded_model_json)
		# load weights into new model
		feature_extractor.load_weights("future_extractor.h5")\

		return feature_extractor

	def save_feature_extractor(self, feature_extractor):

		model_json = feature_extractor.to_json()
		with open("future_extractor.json", "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		feature_extractor.save_weights("future_extractor.h5")
	

	def train_with_randomforest(self, feature_extractor):
		X_for_RF = feature_extractor.predict(self.X_train) 
		RF_model = RandomForestClassifier(n_estimators = 250, random_state = 42)
		RF_model.fit(X_for_RF, self.y_train)

		self.predict_accuracy(feature_extractor, RF_model)

	def predict_accuracy(self, feature_extractor, RF_model):
		X_test_feature = feature_extractor.predict(self.X_test)
		prediction_RF = RF_model.predict(X_test_feature)
		print ("Accuracy = ", metrics.accuracy_score(self.y_test, prediction_RF))




def get_data(folder):
    x = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith("."):
            if folderName in ["nofire"]:
                label = 0
            elif folderName in ["fire"]:
                label = 1
            else:
                label = 2
            for image_filename in tqdm(os.listdir(folder +"/" +folderName+"/")):
                img_file = cv2.imread(folder + "/" +folderName + "/" + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file,(128,128,3), mode = "constant",anti_aliasing=True)
                    img_arr = np.asarray(img_file)
                    x.append(img_arr)
                    y.append(label)
    x = np.asarray(x)
    y = np.asarray(y)
    return x,y

if __name__ == "__main__":

	if os.path.exists("xtrain.npy") and os.path.exists("ytrain.npy") and os.path.exists("xtest.npy") and os.path.exists("ytest.npy"):

		X_train = np.load("xtrain.npy")
		y_train = np.load("ytrain.npy")
		X_test = np.load("xtest.npy")
		y_test = np.load("ytest.npy")

	else:

		train = "./dataset/train"

		test = "./dataset/test"

		X_train, y_train = get_data(train)
		X_test, y_test = get_data(test)

		np.save("xtrain.npy",X_train)
		np.save("ytrain.npy",y_train)
		np.save("xtest.npy",X_test)
		np.save("ytest.npy",y_test)



	fire_detection = FireDetection(X_train, y_train, X_test, y_test)

	if os.path.exists("future_extractor.h5") and os.path.exists("future_extractor.json"):
		
		feature_extractor = fire_detection.load_feature_extractor()
	
	else:
		
		feature_extractor = fire_detection.train_model(batch_size = 8, epochs = 50)

	fire_detection.train_with_randomforest(feature_extractor)









