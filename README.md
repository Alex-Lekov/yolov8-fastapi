# yolov8-fastapi
FastAPI for YOLOv8 (Object Detection) in Docker. This project serves as a template for object detection using YOLOv8 and FastAPI.

### Sample

<img width=600 src="./tests/res/fastapi_sample.png" alt="">

# What's included

- YOLOv8: A popular real-time object detection model
- FastAPI: A modern, fast (high-performance) web framework for building APIs
- Docker: A platform for easily building, shipping, and running distributed applications


---
# Getting Start
Start the application using Docker:
```
docker-compose up
```

Or, start the application locally:

```
pip install -r requirements.txt
```

```
uvicorn main:app --reload --host 0.0.0.0 --port 8001
```  
*Note: You can change the address and port in the file **docker-compose.yaml***

## FAST API Docs url:
http://0.0.0.0:8001/docs#/

<img width=600 src="./tests/res/fastapi.png" alt="FAST API">    

---
# 🚀 Code Examples
### Example 1: Object Detection to JSON   
The following code demonstrates how to perform object detection and receive the results in JSON format:
```python
import requests

input_image_name = 'test_image.jpg'
api_host = 'http://0.0.0.0:8001/'
type_rq = 'img_object_detection_to_json'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

data = response.json()     
print(data)
```
Output:
```
{'detect_objects': [{'name': 'cat', 'confidence': 0.926225245}, {'name': 'dog', 'confidence': 0.9109069705}], 'detect_objects_names': 'cat, dog'}
```

### Example 2: Object Detection to Image    
The following code demonstrates how to perform object detection and receive the results in image format.
```python
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

input_image_name = 'test_image.jpg'
api_host = 'http://0.0.0.0:8001/'
type_rq = 'img_object_detection_to_img'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

img = Image.open(BytesIO(response.content)) 
plt.imshow(img)
```

More examples in the notebook [test.ipynb](./tests/test.ipynb)  

---

# Overview of the code
* [main.py](./main.py) - Base FastAPI functions  
* [app.py](./app.py) - YoloV8 functions     
* [./models](./models) - YoloV8 models folder    

---
# Test
I wrote functional tests for the program to check the operation of the service     

first install the necessary environment or run the docker container
```
pip install -r requirements.txt
```

then just run the tests from the program directory
```
pytest -v --disable-warnings
```

if all is well, you will get the following result:     
<img width=600 src="./tests/res/tests.png" alt="">    

Also added a jupyter notebook with visualization [test.ipynb](./tests/test.ipynb)    


---

# Contact

[Telegram Group](https://t.me/automlalex)