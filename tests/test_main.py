import pytest
import requests
import time
from PIL import Image
import pandas as pd
import sys
import os

# FastAPI
from fastapi.testclient import TestClient
from fastapi import File, HTTPException

# pytest app import fix
dynamic_path = os.path.abspath('.')
print(dynamic_path)

sys.path.append(dynamic_path)

from main import crop_image_by_predict
from main import app

################################ Fixtures #####################################################


@pytest.fixture
def test_image():
    files = {'file': open('./tests/test_image.jpg', 'rb')}
    return(files)

@pytest.fixture
def test_client():
    return TestClient(app)

################################ Test #####################################################


def test_healthcheck(test_client):
    """
    This test function is used to test the /healthcheck endpoint of the application.
    It uses the test client to send a GET request to the endpoint and then asserts that the response has a status code of 200 and a json payload of {"healthcheck": "Everything OK!"}
    This function is important to check if the application is running correctly and all the dependencies are working as expected.
    """
    # Send a GET request to the '/healthcheck' endpoint
    response = test_client.get("/healthcheck")
    # Assert that the response has a status code of 200
    assert response.status_code == 200
    # Assert that the response has a json payload of {"healthcheck": "Everything OK!"}
    assert response.json() == {"healthcheck": "Everything OK!"}


def test_crop_image_by_predict():
    """
    Test for the crop_image_by_predict function.
    This function crops an image given a prediction dataframe and a class name.
    """
    # Create a test image
    test_image = Image.new("RGB", (100, 100), "white")

    # Create a test predict dataframe
    test_predict = pd.DataFrame(
        data={
            "name": ["test_object", "other_object"],
            "xmin": [10, 20],
            "ymin": [10, 20],
            "xmax": [50, 60],
            "ymax": [50, 60],
        }
    )

    # Test cropping by an object that is present in the image
    cropped_image = crop_image_by_predict(test_image, test_predict, "test_object")
    assert cropped_image.size == (40, 40)

    # Test cropping by an object that is not present in the image
    with pytest.raises(HTTPException) as excinfo:
        crop_class_name = "non_existent_object" 
        crop_image_by_predict(test_image, test_predict, "non_existent_object")
    assert excinfo.value.detail == f"{crop_class_name} not found in photo"



def test_img_object_detection_to_json(test_client, test_image):
    """Test the image recognition to json endpoint with a test image.

    Args:
        test_client (TestClient): Fixture that allows to send HTTP requests to the application.
        image_meters_0 (Tuple): Fixture that contains a test image file.
    """
    # Send a POST request to the endpoint with the test image
    response = test_client.post("/img_object_detection_to_json", files=test_image)

    # Assert that the request was successful (status code 200)
    assert response.status_code == 200

    # Assert that the response is in JSON format
    assert response.headers["content-type"] == "application/json"

    # Get the JSON data from the response
    data = response.json()

    # Assert that the JSON data contains the expected keys
    assert 'detect_objects_names' in data
    assert 'detect_objects' in data

    # Assert that the detect_objects values are not None
    assert data['detect_objects_names'] is not None
    assert data['detect_objects'] is not None

    # Assert that the rate values are 'cat, dog'
    assert data['detect_objects_names'] == 'cat, dog'

    # Additional asserts can be added to check for specific values of the rate type
    # or other keys in the JSON data, if needed.



def test_img_object_detection_to_img(test_client, test_image):
    """This test is checking the functionality of the endpoint "/img_object_detection_to_img" using the test_client fixture and the image_meters_0 file. It is performing a POST request to the endpoint with image_meters_0 as the file.
    The test asserts that the response status code is 200, indicating a successful request. It also asserts that the content type of the response is "image/jpeg" and that the content of the response is not None. This indicates that the image was properly returned in the response.
    The test also includes a comment suggesting that additional assertions can be added to check if the image is properly annotated with bounding boxes, which can be used to detect the display on the meter.
    """
    # send POST request to endpoint with image_meters_0 as the file
    response = test_client.post("/img_object_detection_to_img", files=test_image)
    # assert that the response status code is 200
    assert response.status_code == 200
    # assert that the content type of the response is "image/jpeg"
    assert response.headers["content-type"] == "image/jpeg"
    # assert that the content type of the response is not None
    assert response.content != None
    # Additional assertions can be added to check if the image is properly annotated with bbox