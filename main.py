# https://youtu.be/pI0wQbJwIIs
"""
For training, watch videos (202 and 203): 
    https://youtu.be/qB6h5CohLbs
    https://youtu.be/fyZ9Rxpoz2I

The 7 classes of skin cancer lesions included in this dataset are:
Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc) 
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

"""


import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model


def getPrediction(filename): 
    SIZE = 150 #Resize to same size as training images
    img_path = 'static/images/'+filename
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    #Load model
    model1 = load_model('models/model_ResNet50.keras')
    model2=load_model('models/model_ResNet50V2.keras')
    model3=load_model('models/model_ResNet152V2.keras')
    model4=load_model('models/model_MobileNetV2.keras')
    model5=load_model('models/model_Xception.keras')
    model6=load_model('models/model_InceptionResNetV2.keras')

    ideal_weights = [0.4, 0.0, 0.0, 0.3, 0.2, 0.2] # Ideal weights
    
    base_models= [model1, model2, model3, model4, model5, model6]

    preds = [model.predict(tf.expand_dims(img, axis=0) , verbose=0) for model in base_models]
    preds=np.array(preds)
    
    Model_Predictions = np.tensordot(preds, ideal_weights, axes=((0),(0))) # Ensemble model prediction
    # Model_Highest_Prediction = np.argmax(Model_Predictions)


    # img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    # pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    classes = ['pituitary', 'notumor', 'meningioma', 'glioma']
    le = LabelEncoder()
    le.fit(classes)
    #le.inverse_transform([2])
    pred_class = le.inverse_transform([np.argmax(Model_Predictions)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class


#test_prediction =getPrediction('0000.jpg')


