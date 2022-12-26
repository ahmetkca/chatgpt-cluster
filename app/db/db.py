import os
import boto3
import pathlib
from dotenv import load_dotenv
from boto3.resources.base import ServiceResource
from mypy_boto3_dynamodb import DynamoDBServiceResource
from uuid import uuid4

base_dir = pathlib.Path(__file__).parent.parent.parent


load_dotenv(base_dir.joinpath('.env'))


class Config:
    DYNAMODB_REGION_NAME=os.getenv('DYNAMODB_REGION_NAME')
    DYNAMODB_ACCESS_KEY_ID=os.getenv('DYNAMODB_ACCESS_KEY_ID')
    DYNAMODB_SECRET_ACCESS_KEY=os.getenv('DYNAMODB_SECRET_ACCESS_KEY')

def initialize_dynamodb() -> DynamoDBServiceResource:
    ddb = boto3.resource(
            'dynamodb',
            region_name=Config.DYNAMODB_REGION_NAME,
            aws_access_key_id=Config.DYNAMODB_ACCESS_KEY_ID,
            aws_secret_access_key=Config.DYNAMODB_SECRET_ACCESS_KEY
        )
    return ddb

if __name__ == "__main__":
    db: ServiceResource = initialize_dynamodb()

    table = db.Table('chatgpt-cluster-users')

    print(table.scan().get('Items', []))

    table.put_item(Item={
        'uid': str(uuid4()),
        'username': 'test',
        'password': 'fakehasedtest',
        'email': 'abc@gmail.com',
        'first_name': 'test',
        'last_name': 'test',
        'is_active': True,
        'is_superuser': False,
    })

    print(table.scan().get('Items', []))

    print(table.get_item(Key={'uid': 'test'}))

    print(table.get_item(Key={'uid': '355d0fa7-b32a-4936-9943-6039f14de1fa'}))

    

