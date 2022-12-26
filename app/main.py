import asyncio
from datetime import datetime, timedelta
import json
import math
from typing import Any, Dict, List, Optional
from starlette.status import HTTP_401_UNAUTHORIZED
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Security,
    status
)
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
    SecurityScopes
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
import random
import time
import os
import httpx

import pathlib
from dotenv import load_dotenv

base_dir = pathlib.Path(__file__).parent.parent

assert load_dotenv(base_dir.joinpath('.env')) == True

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = os.getenv("SECRET_KEY") 
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
if SECRET_KEY is None or ALGORITHM is None or ACCESS_TOKEN_EXPIRE_MINUTES is None:
    raise Exception("Required environment variables are not set correctly")

ACCESS_TOKEN_EXPIRE_MINUTES = int(ACCESS_TOKEN_EXPIRE_MINUTES)


fake_users_db = {
    "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
}

def fake_hash_password(password: str):
    return "fakehashed" + password

    
def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)



class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None
    scopes: List[str] = []

class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "me": "Read information about the current user",
        "conversation": "Post conversation payload and get response in return",
        "register-service": "Register self maintained chatgpt service",
    }
)


# Seed the generator with the current time in milliseconds
random.seed(int(time.time() * 1000))

def generate_random_number(a: int | None, b: int) -> int:
    if a is None:
        a = 0
    val = random.uniform(a, b)
    val = math.floor(val)
    return val



app = FastAPI()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user


def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
        security_scopes: SecurityScopes, token: str = Depends(oauth2_scheme)
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_scopes = payload.get("scopes", [])
        token_data = TokenData(scopes=token_scopes, username=username)
    except (JWTError, ValidationError):
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
    return user


async def get_current_active_user(
    current_user: User = Security(get_current_user, scopes=["me"])
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
            data={"sub": user.username, "scopes": form_data.scopes}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


class ServiceRegistrationPayload(BaseModel):
    domainName: str
    port: int
    checkAuthSessionEndpoint: str
    conversationEndpoint: str
    sendAuthReminderRequestEndpoint: str | None



class ConversationPayload(BaseModel):
    class ConversationMessage(BaseModel):
        class ConversationMessageContent(BaseModel):
            content_type: str
            parts: list[str]
        id: str
        role: str
        content: ConversationMessageContent

    action: str
    conversation_id: str | None
    messages: list[ConversationMessage]
    parent_message_id: str
    model: str
    callback_url: str | None


class ServiceKV:
    def __init__(self, kv: Dict[str, Dict[str, Any]] = {}) -> None:
        self.__services: Dict[str, Dict[str, Any]] = kv
        pass

    def Put(self, domainName: str, info: Dict[str, Any]) -> Optional[str]:
        if domainName in self.__services:
            return None

        self.__services[domainName] = info

        return domainName

    def Get(self, domainName) -> Optional[Dict[str, Any]]:
        if domainName in self.__services:
            return self.__services[domainName]

        return None

    def GetAll(self):
        return {k: v for k, v in self.__services.items()}

    def Delete(self, domainName: str) -> Optional[str]:
        if domainName in self.__services:
            del self.__services[domainName]
            return domainName

        return None
    
    # O(1) time complexity
    def GetRandomService(self) -> Optional[Dict[str, Any]]:
        if len(self.__services) == 0:
            return
        
        keys = list(self.__services.keys())

        return self.__services[keys[generate_random_number(None, len(keys))]]



serviceKV = ServiceKV()


@app.get("/")
async def home():
    return {
        "ChatGPT Cluster": "v0.0.1",
        "description": "ChatGPT Cluster is a service that allows you to register your ChatGPT service and get a list of all registered services.",
        "total_services": len(serviceKV.GetAll()),
        "register": "POST /register",
        "get_all_services": "GET /services",
        "get_service": "GET /services/{domainName}",
    }


@app.post("/register")
async def register(serviceInfo: ServiceRegistrationPayload):
    dct = json.loads(serviceInfo.json())
    print(dct)
    return serviceKV.Put(serviceInfo.domainName, dct)


@app.get("/services")
async def get_all_services():
    for service in serviceKV.GetAll():
        print(service)
    return [{service[0]: service[1]} for service in serviceKV.GetAll().items()]

@app.get("/services/{domainName}")
async def get_service(domainName: str):
    return serviceKV.Get(domainName)


async def request_conversation_endpoint(service: Dict[str, Any], payload: ConversationPayload) -> Optional[httpx.Response]:
    if service.get("conversationEndpoint") is None:
        return None
    if service.get("conversationEndpoint") == "":
        return None
    if service.get("domainName") is None:
        return None
    if service.get("domainName") == "":
        return None
    url = service.get("domainName") + service.get("conversationEndpoint")
    await asyncio.sleep(3)
    print("Making request to: " + url)
    print(json.dumps(dict(payload), indent=4, sort_keys=True, default=str, ensure_ascii=False, allow_nan=True))

    return None
    # async with httpx.AsyncClient() as client:
    #     response = await client.post(url, json=payload.dict())
    #     return response

async def makeRequestToService(service: Dict[str, Any], payload: ConversationPayload) -> None:
    print("Making request to service: " + service["domainName"])
    print(json.dumps(dict(payload), indent=4, sort_keys=True, default=str, ensure_ascii=False, allow_nan=True))
    response = await request_conversation_endpoint(service, payload)
    if response is None:
        return None
    with httpx.AsyncClient() as client:
        response = await client.post(payload.callback_url, json=response.json())
        print(response.json())

from sse_starlette.sse import EventSourceResponse

@app.post("/conversation")
async def conversation(
    conversationPayload: ConversationPayload, 
    background_tasks: BackgroundTasks,
    current_user: User = Security(get_current_active_user, scopes=["conversation"])
):
    print(json.dumps(dict(conversationPayload), indent=4, sort_keys=True, default=str, ensure_ascii=False, allow_nan=True))
    # TODO: Get a random service and send a request to it
    # TODO: Return the response from the service
    # TODO: Each request will have default amount of retries, default retry TBD
    # TODO: If the request fails, it will retry with another service
    # TODO: Provide two type of requests, one synchronous and one asynchronous
    # TODO: Synchronous request will wait for the response from the service
    # TODO: Asynchronous request will return the response immediately and will retry in the background to a callback URL
    service = serviceKV.GetRandomService()

    if service is None:
        raise HTTPException(status_code=404, detail="No registered chatgpt services found")

    if conversationPayload.callback_url is not None:
        print("Asynchronous request")
        background_tasks.add_task(makeRequestToService, service, conversationPayload)
        return {"status": "conversation request sent to service"}
    
    print("Synchronous request")
    async def retry_request():
        retry = 3
        while retry > 0:
            response = await request_conversation_endpoint(service, conversationPayload)
            if response is None:
                yield {
                    'event': 'conversation_response',
                    'id': f'retry#-{retry}',
                    'data': {
                        "status": f"Failed to make a request to the service, retrying... {retry}",
                    }
                }
                retry -= 1
                continue
            if response.status_code == 200:
                yield {
                    'event': 'conversation_response',
                    'id': retry,
                    'data': response.json()
                }
            yield {
                'event': 'conversation_response',
                'id': retry,
                'data': {
                    "status": f"Failed to make a request to the service, retrying... {retry}",
                }
            }
            retry -= 1
    return EventSourceResponse(retry_request())
    # raise HTTPException(status_code=500, detail="Failed to make a request to the service")
    
    



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7070, reload=True, workers=1)
