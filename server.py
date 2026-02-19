import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, json
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder
import torch
import base64
from io import BytesIO
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_name_or_path = "/root/qwen3-vl/Qwen3-VL-Embedding-8B"
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)

def base64_to_image(image_base64: str):
    image_data = base64.b64decode(image_base64)
    image_file = BytesIO(image_data)
    return Image.open(image_file)

def get_features(text, image):
    input = {}
    if text:
        input["text"] = text
    if image:
        input["image"] = image
    return model.process([input])[0].tolist()

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    json_post_raw = await request.json()
    messages = json_post_raw
    text = messages.get('text')
    image = messages.get('image')
    if not text and not image:
        return {"code": 400, "message": "text 和 image 至少传一个！"}
    t = time.time()
    if image:
        image = base64_to_image(image)
    embeds = get_features(text, image)
    return {
        "code": 200,
        "message": "success",
        "data": embeds,
        "use_time": time.time() - t
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8848, workers=1)