---
license: apache-2.0
library_name: transformers
pipeline_tag: feature-extraction

base_model:
- Qwen/Qwen3-VL-8B-Instruct
tags:
- transformers
- multimodal embedding
- qwen
- embedding
---

# Qwen3-VL-Embedding-8B

<p align="center">
    <img src="https://model-demo.oss-cn-hangzhou.aliyuncs.com/Qwen3-VL-Embedding.png" width="400"/>
<p>

## Highlights

The **Qwen3-VL-Embedding** and **Qwen3-VL-Reranker** model series are the latest additions to the Qwen family, built upon the recently open-sourced and powerful Qwen3-VL foundation model. Specifically designed for multimodal information retrieval and cross-modal understanding, this suite accepts diverse inputs including text, images, screenshots, and videos, as well as inputs containing a mixture of these modalities.

While the Embedding model generates high-dimensional vectors for broad applications like retrieval and clustering, the Reranker model is engineered to refine these results, establishing a comprehensive pipeline for state-of-the-art multimodal search.

- **Multimodal Versatility**: Both models seamlessly handle a wide range of inputs—including text, images, screenshots, and video—within a unified framework. They deliver state-of-the-art performance across diverse multimodal tasks such as image-text retrieval, video-text matching, visual question answering (VQA), and multimodal content clustering.

- **Unified Representation Learning (Embedding)**: By leveraging the Qwen3-VL architecture, the Embedding model generates semantically rich vectors that capture both visual and textual information in a shared space. This facilitates efficient similarity computation and retrieval across different modalities.

- **High-Precision Reranking (Reranker)**: We also introduce the Qwen3-VL-Reranker series to complement the embedding model. The reranker takes a (query, document) pair as input—where both query and document may contain arbitrary single or mixed modalities—and outputs a precise relevance score. In retrieval pipelines, the two models are typically used in tandem: the embedding model performs efficient initial recall, while the reranker refines results in a subsequent re-ranking stage. This two-stage approach significantly boosts retrieval accuracy.

- **Exceptional Practicality**: Inheriting Qwen3-VL’s multilingual capabilities, the series supports over 30 languages, making it ideal for global applications. It is highly practical for real-world scenarios, offering flexible vector dimensions, customizable instructions for specific use cases, and strong performance even with quantized embeddings. These capabilities enable developers to seamlessly integrate both models into existing pipelines, unlocking powerful cross-lingual and cross-modal understanding.

## Model Overview

**Qwen3-VL-Embedding-8B** has the following features:

- Model Type: MultiModal Embedding
- Supported Languages: 30+ Languages
- Supported Input Modalities: Text, images, screenshots, videos, and arbitrary multimodal combinations (e.g., text + image, text + video)
- Number of Parameters: 8B
- Context Length: 32k
- Embedding Dimension: Up to 4096, supports user-defined output dimensions ranging from 64 to 4096

For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our [technical report](https://arxiv.org/abs/2601.04720), [blog](https://qwen.ai/blog?id=qwen3-vl-embedding), [GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding).

## Qwen3-VL-Embedding and Qwen3-VL-Reranker Model list

| Model | Size | Model Layers | Sequence Length | Embedding Dimension | Quantization Support | MRL Support | Instruction Aware |
|---|---|---|---|---|----------------------|---|---|
| [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | 2B | 28 | 32K | 2048 | Yes | Yes | Yes |
| [Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) | 8B | 36 | 32K | 4096 | Yes | Yes | Yes |
| [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | 2B | 28 | 32K | - | -  | - | Yes |
| [Qwen3-VL-Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) | 8B | 36 | 32K | - | -  | - | Yes |

> **Note**:
> - `Quantization Support` indicates the supported quantization post process for the output embedding. 
> - `MRL Support` indicates whether the embedding model supports custom dimensions for the final embedding. 
> - `Instruction Aware` notes whether the embedding or reranking model supports customizing the input instruction according to different tasks.
> Our evaluation indicates that, for most downstream tasks, using instructions (instruct) typically yields an improvement of 1% to 5% compared to not using them. Therefore, we recommend that developers create tailored instructions specific to their tasks and scenarios. In multilingual contexts, we also advise users to write their instructions in English, as most instructions utilized during the model training process were originally written in English.

## Model Performance

### Evaluation Results on [MMEB-V2](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard)

Results on the MMEB-V2 benchmark. All models except IFM-TTE have been re-evaluated on the updated VisDoc OOD split. CLS: classification, QA: question answering, RET: retrieval, GD: grounding, MRET: moment retrieval, VDR: ViDoRe, VR: VisRAG, OOD: out-of-distribution.

| Model                      | Model Size | Image CLS | Image QA | Image RET | Image GD | Image Overall | Video CLS | Video QA | Video RET | Video MRET | Video Overall | VisDoc VDRv1 | VisDoc VDRv2 | VisDoc VR | VisDoc OOD | VisDoc Overall | All    |
|----------------------------|---------|-------|------|------|------|-----------|------|------|------|------|------|-------|------|--------|------|------|--------|
| **# of Datasets →**        |         | 10    | 10   | 12   | 4    | 36        | 5    | 5    | 5    | 3    | 18   | 10    | 4    | 6      | 4    | 24   | 78     |
| VLM2Vec                    | 2B      | 58.7 | 49.3 | 65.0 | 72.9 | 59.7 | 33.4 | 30.5 | 20.6 | 30.7 | 28.6 | 49.8 | 13.5 | 51.8 | 48.2 | 44.0 | 47.7 |
| VLM2Vec-V2                 | 2B      | 62.9 | 56.3 | 69.5 | 77.3 | 64.9 | 39.3 | 34.3 | 28.8 | 36.8 | 34.6 | 75.5 | 44.9 | 79.4 | 62.2 | 69.2 | 59.2 |
| GME-2B                     | 2B      | 54.4 | 29.9 | 66.9 | 55.5 | 51.9 | 34.9 | 42.0 | 25.6 | 31.1 | 33.6 | 86.1 | 54.0 | 82.5 | 67.5 | 76.8 | 55.3 |
| GME-7B                     | 7B      | 57.7 | 34.7 | 71.2 | 59.3 | 56.0 | 37.4 | 50.4 | 28.4 | 37.0 | 38.4 | 89.4 | 55.6 | 85.0 | 68.3 | 79.3 | 59.1 |
| Ops-MM-embedding-v1        | 8B      | 69.7 | 69.6 | 73.1 | 87.2 | 72.7 | 59.7 | 62.2 | 45.7 | 43.2 | 53.8 | 80.1 | 59.6 | 79.3 | 67.8 | 74.4 | 68.9 |
| IFM-TTE                    | 8B      | 76.7 | 78.5 | 74.6 | 89.3 | 77.9 | 60.5 | 67.9 | 51.7 | 54.9 | 59.2 | 85.2 | 71.5 | 92.7 | 53.3 | 79.5 | 74.1 |
| RzenEmbed                  | 8B      | 70.6 | 71.7 | 78.5 | 92.1 | 75.9 | 58.8 | 63.5 | 51.0 | 45.5 | 55.7 | 89.7 | 60.7 | 88.7 | 69.9 | 81.3 | 72.9 |
| Seed-1.6-embedding-1215    | unknown | 75.0 | 74.9 | 79.3 | 89.0 | 78.0 | 85.2 | 66.7 | 59.1 | 54.8 | 67.7 | 90.0 | 60.3 | 90.0 | 70.7 | 82.2 | 76.9 | 
| **Qwen3-VL-Embedding-2B**  | 2B      | 70.2 | 74.4 | 74.9 | 88.6 | 75.0 | 72.8 | 63.8 | 52.3 | 51.6 | 61.1 | 85.2 | 66.0 | 86.3 | 74.3 | 80.2 | 73.4 |
| **Qwen3-VL-Embedding-8B**  | 8B      | 74.4 | 81.0 | 80.0 | 92.2 | 80.1 | 79.1 | 70.1 | 57.0 | 53.2 | 66.1 | 88.2 | 69.9 | 88.8 | 78.3 | 83.3 | **77.9** |

### Evaluation Results on [MMTEB](https://huggingface.co/spaces/mteb/leaderboard)

Results on the MMTEB benchmark. 

| Model                            |  Size   |  Mean (Task)  | Mean (Type) | Bitxt Mining | Class. | Clust. | Inst. Retri. | Multi. Class. | Pair. Class. | Rerank | Retri. | STS  |
|----------------------------------|:-------:|:-------------:|:-------------:|:--------------:|:--------:|:--------:|:--------------:|:---------------:|:--------------:|:--------:|:--------:|:------:|
| NV-Embed-v2                      |   7B    |     56.29     | 49.58       | 57.84        | 57.29  | 40.80  | 1.04         | 18.63         | 78.94        | 63.82  | 56.72  | 71.10|
| GritLM-7B                        |   7B    |     60.92     | 53.74       | 70.53        | 61.83  | 49.75  | 3.45         | 22.77         | 79.94        | 63.78  | 58.31  | 73.33|
| BGE-M3                           |  0.6B   |     59.56     | 52.18       | 79.11        | 60.35  | 40.88  | -3.11        | 20.1          | 80.76        | 62.79  | 54.60  | 74.12|
| multilingual-e5-large-instruct   |  0.6B   |     63.22     | 55.08       | 80.13        | 64.94  | 50.75  | -0.40        | 22.91         | 80.86        | 62.61  | 57.12  | 76.81|
| gte-Qwen2-1.5B-instruct          |  1.5B   |     59.45     | 52.69       | 62.51        | 58.32  | 52.05  | 0.74         | 24.02         | 81.58        | 62.58  | 60.78  | 71.61|
| gte-Qwen2-7b-Instruct            |   7B    |     62.51     | 55.93       | 73.92        | 61.55  | 52.77  | 4.94         | 25.48         | 85.13        | 65.55  | 60.08  | 73.98|
| text-embedding-3-large           |    -    |     58.93     | 51.41       | 62.17        | 60.27  | 46.89  | -2.68        | 22.03         | 79.17        | 63.89  | 59.27  | 71.68|
| Cohere-embed-multilingual-v3.0   |    -    |     61.12     | 53.23       | 70.50        | 62.95  | 46.89  | -1.89        | 22.74         | 79.88        | 64.07  | 59.16  | 74.80|
| Gemini Embedding                 |    -    |     68.37     | 59.59       | 79.28        | 71.82  | 54.59  | 5.18         | **29.16**     | 83.63        | 65.58  | 67.71  | 79.40|
| Qwen3-Embedding-0.6B        |  0.6B   |     64.33     | 56.00       | 72.22        | 66.83  | 52.33  | 5.09         | 24.59         | 80.83        | 61.41  | 64.64  | 76.17|
| Qwen3-Embedding-4B           |   4B    |     69.45     | 60.86       | 79.36        | 72.33  | 57.15  | **11.56**    | 26.77         | 85.05        | 65.08  | 69.60  | 80.86|
| Qwen3-Embedding-8B          |   8B    |   **70.58**   | **61.69**   | **80.89**    | **74.00** | **57.65** | 10.06      | 28.66         | **86.40**    | **65.63** | **70.88** | **81.08** |
| Qwen3-VL-Embedding-2B | 2B | 63.87 | 55.84 | 69.51 | 65.86 | 52.50 | 3.87 | 26.08 | 78.50 | 64.80 | 67.12 | 74.29 |
| Qwen3-VL-Embedding-8B | 8B | 67.88 | 58.88 | 77.48 | 71.95 | 55.82 | 4.46 | 28.59 | 81.08 | 65.72 | 69.41 | 75.41 |


## Usage

- **requirements**
```text
transformers>=4.57.0
qwen-vl-utils>=0.0.14
torch==2.8.0
```

### Basic Usage Example

```python
from scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# Define a list of query texts
queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."}
]

# Define a list of document texts and images
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
]

# Specify the model path
model_name_or_path = "Qwen/Qwen3-VL-Embedding-8B"

# Initialize the Qwen3VLEmbedder model
model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)

# Compute similarity scores between query embeddings and document embeddings
similarity_scores = (embeddings[:4] @ embeddings[4:].T)

# Print out the similarity scores in a list format
print(similarity_scores.tolist())

# [[0.74267578125, 0.6630859375, 0.6328125], [0.443603515625, 0.33349609375, 0.396484375], [0.3671875, 0.2354736328125, 0.289306640625], [0.060821533203125, -0.01557159423828125, 0.0165863037109375]]
```

### vLLM Basic Usage Example
```python
import argparse
import numpy as np
import os
from typing import List, Dict, Any
from vllm import LLM, EngineArgs
from vllm.multimodal.utils import fetch_image


# Define a list of query texts
queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."}
]

# Define a list of document texts and images
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
]

def format_input_to_conversation(input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> List[Dict]:
    content = []
    
    text = input_dict.get('text')
    image = input_dict.get('image')
    
    if image:
        image_content = None
        if isinstance(image, str):
            if image.startswith(('http', 'https', 'oss')):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = 'file://' + abs_image_path
        else:
            image_content = image
        
        if image_content:
            content.append({
                'type': 'image', 
                'image': image_content,
            })
    
    if text:
        content.append({'type': 'text', 'text': text})
    
    if not content:
        content.append({'type': 'text', 'text': ""})
    
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content}
    ]
    
    return conversation

def prepare_vllm_inputs(input_dict: Dict[str, Any], llm, instruction: str = "Represent the user's input.") -> Dict[str, Any]:
    text = input_dict.get('text')
    image = input_dict.get('image')
    
    conversation = format_input_to_conversation(input_dict, instruction)
    
    prompt_text = llm.llm_engine.tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    multi_modal_data = None
    if image:
        if isinstance(image, str):
            if image.startswith(('http', 'https', 'oss')):
                try:
                    image_obj = fetch_image(image)
                    multi_modal_data = {"image": image_obj}
                except Exception as e:
                    print(f"Warning: Failed to fetch image {image}: {e}")
            else:
                abs_image_path = os.path.abspath(image)
                if os.path.exists(abs_image_path):
                    from PIL import Image
                    image_obj = Image.open(abs_image_path)
                    multi_modal_data = {"image": image_obj}
                else:
                    print(f"Warning: Image file not found: {abs_image_path}")
        else:
            multi_modal_data = {"image": image}
    
    result = {
        "prompt": prompt_text,
        "multi_modal_data": multi_modal_data
    }
    return result

def main():
    parser = argparse.ArgumentParser(description="Offline Similarity Check with vLLM")
    parser.add_argument("--model-path", type=str, default="models/Qwen3-VL-Embedding-8B", help="Path to the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (e.g., bfloat16)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    engine_args = EngineArgs(
        model=args.model_path,
        runner="pooling",
        dtype=args.dtype,
        trust_remote_code=True,
    )
    
    llm = LLM(**vars(engine_args))
    
    all_inputs = queries + documents
    vllm_inputs = [prepare_vllm_inputs(inp, llm) for inp in all_inputs]
    
    
    outputs = llm.embed(vllm_inputs)
    
    embeddings_list = []
    for i, output in enumerate(outputs):
        emb = output.outputs.embedding
        embeddings_list.append(emb)
        print(f"Input {i} embedding shape: {len(emb)}")
    
    embeddings = np.array(embeddings_list)
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    num_queries = len(queries)
    query_embeddings = embeddings[:num_queries]
    doc_embeddings = embeddings[num_queries:]
    
    similarity_scores = query_embeddings @ doc_embeddings.T
    
    print("\nSimilarity Scores:")
    print(similarity_scores.tolist())
    

if __name__ == "__main__":
    main()
```

### SGLang Basic Usage Example
```python
import argparse
import numpy as np
import torch
import os
from typing import List, Dict, Any
from sglang.srt.entrypoints.engine import Engine

# Define a list of query texts
queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."}
]

# Define a list of document texts and images
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}
]

def format_input_to_conversation(input_dict: Dict[str, Any], instruction: str = "Represent the user's input.") -> List[Dict]:
    content = []
    
    text = input_dict.get('text')
    image = input_dict.get('image')

    if image:
        image_content = None
        if isinstance(image, str):
            if image.startswith(('http', 'oss')):
                image_content = image
            else:
                abs_image_path = os.path.abspath(image)
                image_content = 'file://' + abs_image_path
        else:
            image_content = image
        if image_content:
            content.append({
                'type': 'image', 'image': image_content,
            })

    if text:
        content.append({'type': 'text', 'text': text})

    if not content:
        content.append({'type': 'text', 'text': ""})

    conversation = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content}
    ]

    return conversation

def convert_to_sglang_format(input_dict: Dict[str, Any], engine: Engine, instruction: str = "Represent the user's input.") -> Dict[str, Any]:
    conversation = format_input_to_conversation(input_dict, instruction)
    
    text_for_api = engine.tokenizer_manager.tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=True
    )

    result = {"text": text_for_api}
    
    image = input_dict.get('image')
    if image and isinstance(image, str):
        result["image"] = image
        
        
    return result

def main():
    parser = argparse.ArgumentParser(description="Offline Similarity Check with SGLang")
    parser.add_argument("--model-path", type=str, default="models/Qwen3-VL-Embedding-8B", help="Path to the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (e.g., bfloat16)")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    
    engine = Engine(
        model_path=args.model_path,
        is_embedding=True,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    inputs = queries + documents
    sglang_inputs = [convert_to_sglang_format(inp, engine) for inp in inputs]
    print(sglang_inputs[:])
    print(f"sglang_inputs: {sglang_inputs}")
    print(f"Processing {len(sglang_inputs)} inputs...")

    prompts = [inp['text'] for inp in sglang_inputs]
    images = [inp.get('image') for inp in sglang_inputs]


    results = engine.encode(prompts, image_data=images)
    
    embeddings_list = []
    for res in results:
        embeddings_list.append(res['embedding'])
            
    embeddings = np.array(embeddings_list)
    print(f"Embeddings shape: {embeddings.shape}")

    num_queries = len(queries)
    query_embeddings = embeddings[:num_queries]
    doc_embeddings = embeddings[num_queries:]
    
    similarity_scores = (query_embeddings @ doc_embeddings.T)

    print("\nSimilarity Scores:")
    print(similarity_scores.tolist())

if __name__ == "__main__":
    main()
```

For more usage examples, please visit our [GitHub repository](https://github.com/QwenLM/Qwen3-VL-Embedding).


## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen3vlembedding,
  title={Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking},
  author={Li, Mingxin and Zhang, Yanzhao and Long, Dingkun and Chen Keqin and Song, Sibo and Bai, Shuai and Yang, Zhibo and Xie, Pengjun and Yang, An and Liu, Dayiheng and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2601.04720},
  year={2026}
}
```