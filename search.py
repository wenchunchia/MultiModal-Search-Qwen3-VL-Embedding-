import requests
from pymilvus import MilvusClient
import matplotlib.pyplot as plt
from PIL import Image
import os

# 绕过系统代理
session = requests.Session()
session.trust_env = False

# 全部使用绝对路径
DB_PATH    = "/root/qwen3-vl/Qwen3-VL-Embedding-8B/milvus_qwen3vl.db"
IMAGE_DIR  = "/root/qwen3-vl/Qwen3-VL-Embedding-8B/images"
COLLECTION = "qwen3_vl_images"
API_URL    = "http://127.0.0.1:8848/v1/embeddings"

client = MilvusClient(DB_PATH)


def embed_text(text):
    resp = session.post(API_URL, json={"text": text}, timeout=30)
    return resp.json()["data"]


def search(query, top_k=1):
    vec = embed_text(query)
    results = client.search(
        collection_name=COLLECTION,
        data=[vec],
        limit=top_k,
        output_fields=["filename"]
    )
    return results[0]


def show_results(query, results):
    valid = []
    for item in results:
        fname = item["entity"]["filename"]
        img_path = os.path.join(IMAGE_DIR, fname)
        print(f"  尝试打开: {img_path}  存在={os.path.exists(img_path)}")
        try:
            img = Image.open(img_path).convert("RGB")
            valid.append((item, img))
        except Exception as e:
            print(f"  ⚠️ 打开失败: {fname} -> {e}")

    if not valid:
        print("没有可显示的图片")
        return

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Query: {query}", fontsize=14, fontweight='bold')
    for i, (item, img) in enumerate(valid):
        fname = item["entity"]["filename"]
        score = item["distance"]
        axes[i].imshow(img)
        axes[i].set_title(f"score: {score:.4f}\n{fname}", fontsize=8)
        axes[i].axis('off')

    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.tight_layout()
    safe_query = query[:20].replace(' ', '_').replace('/', '_')
    out_path = f"/root/qwen3-vl/Qwen3-VL-Embedding-8B/result_{safe_query}.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ 结果已保存: {out_path}")
    plt.close()


def main():
    # 启动时验证
    print(f"数据库: {DB_PATH}  存在={os.path.exists(DB_PATH)}")
    print(f"图片目录: {IMAGE_DIR}  存在={os.path.exists(IMAGE_DIR)}")
    cols = client.list_collections()
    print(f"Collections: {cols}")

    # 取一条数据验证
    sample = client.search(COLLECTION, data=[[0.0]*4096], limit=1, output_fields=["filename"])
    print(f"样本数据: {sample[0][0]}")

    print("\n=== 文搜图检索系统 (输入 q 退出) ===\n")
    while True:
        query = input("请输入搜索词: ").strip()
        if not query:
            continue
        if query == "q":
            break
        print("检索中...")
        results = search(query, top_k=3)
        print(f"Top {len(results)} 结果：")
        for i, r in enumerate(results):
            print(f"  {i+1}. {r['entity']['filename']}  score={r['distance']:.4f}")
        show_results(query, results)


if __name__ == '__main__':
    main()
