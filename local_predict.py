import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('vgg16_scratch.h5')
test_image = image.load_img('beans.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print (result)

y_pred = np.argmax(result,axis=1)
print (y_pred[0])

output_labels = ['Bean','Brinjal','Broccoli','Cabbage','Capsicum','Carrot','Cauliflower','Cucumber','Papaya','Potato','Radish','Tomato']
print(output_labels[y_pred[0]])
