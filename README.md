# PRM inference

## huggingface inference

```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/path/to/prm_model/", trust_remote_code=True)
model = AutoModel.from_pretrained("/path/to/prm_model/", device_map="auto", trust_remote_code=True).eval()

inputs = tokenizer(['陕西的省会是西安', "介绍一下大熊猫"], return_tensors='pt', padding=True).to(model.device)
response = model(**inputs)
print(response.logits)
```

## vllm server for inference

1. install vllm and install vllm prm plugin
```shell
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
pip install vllm==v0.6.4.post1
cd skywork-o1-prm-inference
pip install -e .
```

2. start vllm server
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/prm_model \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto
```

3. request server for inference

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id
responses = client.embeddings.create(
    input=["你是谁?", "你好"],
    model=model,
)
scores = [data.embedding[-1] for data in responses.data]
print(scores)
```