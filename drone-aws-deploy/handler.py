try:
    import unzip_requirements
except importError:
    pass
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import PIL

import boto3
import os
import io
import json
import base64
from requests_toolbelt.multipart import decoder

print("Import End...")

#define environment variables if there are not exist
S3_BUCKET   =   os.environ['S3_BUCKET'] #if 'S3_BUCKET' in os.environ else 'tsai-models-s1'
MODEL_PATH  =   os.environ['MODEL_PATH']#if 'MODEL_PATH' in os.environ else 'resnet34.pt' 

class_names = ['Flying Birds', 'Large QuadCopters', 'Small QuadCopters', 'Winged Drones']

print('Downloading model...')

s3 = boto3.client('s3')

try:
    if os.path.isfile(MODEL_PATH) != True:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        print("Creating Bytestream")
        bytestream = io.BytesIO(obj['Body'].read())
        print('Loading Model')
        model = torch.jit.load(bytestream)
        print('Model Loaded...')
except Exception as e:
    print(repr(e))
    raise(e)

def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize( (224,224), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5270, 0.5794, 0.6113], std=[0.1725, 0.1665, 0.1815])])
        image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    return model(tensor).argmax().item()

def classify_image(event, context):
    try:
        content_type_header = event['headers']['content-type']
        print(event['body'])
        body = base64.b64decode(event['body'])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        prediction = get_prediction(image_bytes=picture.content)
        print(prediction)

        prediction_label = class_names[prediction]

        filename = picture.headers[b'content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'content-Disposition'].decode().split(';')[2].split('=')[1]

        return {
            "statusCode": 200,
            "headers": {
                'content-type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True

            },
            "body": json.dumps({'file':filename.replace('"',''), 'predicted':f"{prediction} : {prediction_label}"})
        }
    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            "headers": {
                'content-type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                "Access-Control-Allow-Credentials": True
            },
            "body": json.dumps({"error: repr(e"})
        }