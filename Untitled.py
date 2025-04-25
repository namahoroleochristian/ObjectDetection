{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e39f8c1a-6023-4c68-b7ac-9b351d5f743b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "cow  :  83.23521614074707\n"
     ]
    }
   ],
   "source": [
    "from imageai.Detection import ObjectDetection\n",
    "import os\n",
    "\n",
    "execution_path = os.getcwd()\n",
    "\n",
    "detector = ObjectDetection()\n",
    "detector.setModelTypeAsTinyYOLOv3()\n",
    "detector.setModelPath(os.path.join(execution_path, \"yolo-tiny.h5\"))\n",
    "detector.loadModel()\n",
    "\n",
    "detections = detector.detectObjectsFromImage(\n",
    "    input_image=os.path.join(execution_path, \"image.jpg\"),\n",
    "    output_image_path=os.path.join(execution_path, \"cImage.jpg\")\n",
    ")\n",
    "\n",
    "for eachObject in detections:\n",
    "    print(eachObject[\"name\"], \" : \", eachObject[\"percentage_probability\"])\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
