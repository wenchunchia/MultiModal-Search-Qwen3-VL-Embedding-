import os
import base64
import requests
from pymilvus import MilvusClient, DataType

# 绕过系统代理
session = requests.Session()
session.trust_env = False

DB_PATH = "/root/qwen3-vl/Qwen3-VL-Embedding-8B/milvus_qwen3vl.db"
client = MilvusClient(DB_PATH)
collection_name = "qwen3_vl_images"
IMAGE_DIR = "/root/qwen3-vl/Qwen3-VL-Embedding-8B/images"


def create_collection():
    if collection_name in client.list_collections():
        print("Collection 已存在，继续追加入库")
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=500)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=4096)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=500)
    schema.verify()
    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    print("Collection 创建成功")


def get_done_set():
    total = client.get_collection_stats(collection_name)["row_count"]
    if total == 0:
        return set()
    rows = client.query(collection_name, filter="id != ''", output_fields=["id"], limit=total)
    return {row["id"] for row in rows}


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def embed_image(image_path):
    image_b64 = encode_image(image_path)
    response = session.post(
        "http://127.0.0.1:8848/v1/embeddings",
        json={"image": image_b64},
        timeout=60
    )
    return response.json()["data"]


def main():
    create_collection()
    done = get_done_set()
    print(f"已入库: {len(done)} 张，继续处理剩余图片...")

    all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = [f for f in all_images if f not in done]
    total = len(images)
    print(f"待处理: {total} 张")

    batch = []
    for i, img_name in enumerate(images):
        img_path = os.path.join(IMAGE_DIR, img_name)
        try:
            vec = embed_image(img_path)
            batch.append({"id": img_name, "vector": vec, "filename": img_name})
            if len(batch) >= 10:
                client.upsert(collection_name=collection_name, data=batch)
                batch = []
            if (i + 1) % 50 == 0:
                print(f"进度: {i+1}/{total} ({img_name})")
        except Exception as e:
            print(f"跳过 {img_name}: {e}")

    if batch:
        client.upsert(collection_name=collection_name, data=batch)
    print(f"✅ 入库完成！共处理 {total} 张图片")


if __name__ == '__main__':
    main()
