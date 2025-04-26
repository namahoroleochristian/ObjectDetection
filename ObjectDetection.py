from imageai.Detection import ObjectDetection
from Ipython.display import Image


import os 
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
detections = detector.detectObjectFromImage( input_image = os.path
                                            .join(execution_path,"image.jpg"),output_image = os.path
                                            .join(execution_path,"NewImage.jpg"))
for Object in detections:
    print(Object["name"],":",Object["percentage_probability"])


Image(filename = "NewImage.jpg")
