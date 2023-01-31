# yolov8-fastapi
FastAPI for YOLOv8 (Object Detection) in Docker. Project template.

### Sample

<img width=600 src="./tests/res/fastapi_sample.png" alt="">

## What's inside:
- YOLOv8
- FastAPI
- Docker
- Python
- PyTorch


---
# Getting Start
```
docker-compose up
```

## FAST API Docs url:
http://0.0.0.0:8008/docs#/

<img width=600 src="./tests/res/fastapi.png" alt="FAST API">    

---
# 🚀 Code Examples
start the service before starting the tests
```uvicorn main:app --reload --host 0.0.0.0 --port 8008```
*you can change the address and port in the file **docker-compose.yaml***
### To value
```python
import requests

input_image_name = 'test_image.jpg'
api_host = 'http://0.0.0.0:8008/'
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

### To Image
```python
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

input_image_name = 'test_image.jpg'
api_host = 'http://0.0.0.0:8008/'
type_rq = 'img_object_detection_to_img'

files = {'file': open(input_image_name, 'rb')}

response = requests.post(api_host+type_rq, files=files)

img = Image.open(BytesIO(response.content)) 
plt.imshow(img)
```

More examples in the notebook [test.ipynb](./tests/test.ipynb)  

---

# Overview of the code
[main.py](./main.py) Base FASTAPI functions    
[app.py](./app.py) YoloV8 functions     
[./models](./models) YoloV8 models folder    

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