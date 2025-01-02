from typing import List
from contextlib import asynccontextmanager
from .topic_moderator.classify import classifyText, classifyTextBatch, loadModel
from fastapi import FastAPI, Body, Response
from io import BytesIO
from TTS.api import TTS
import torch
import scipy
import numpy as np

model = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    model = loadModel()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/classify")
def classify(text: str = Body()):
    return classifyText(model, text)


@app.post("/batch_classify")
def batchClassify(texts: List[str] = Body()):
    return classifyTextBatch(model, *texts)
