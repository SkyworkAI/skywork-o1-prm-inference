<p align="center">
<h1>
  ✨Skywork-o1-prm-inference
</h1>
<p align="center">
🤗 <a href="https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B" target="_blank">Hugging Face</a> • 🤖 <a href="https://modelscope.cn/organization/Skywork" target="_blank">ModelScope</a>
</p>

<div align="center">
<h4 align="center">
    <p>
        <a href="https://github.com/SkyworkAI/skywork-o1-prm-inference/blob/main/README.md">English</a> |
        <b>中文</b>
    <p>
</h4>

</div>

本项目为使用Skywork-o1-prm模型的用户提供推理部署指南。

# PRM模型推理

## 下载地址
| 模型 | 下载地址 | 描述 |
| --------- | ------ | ------ | 
| Skywork-o1-Open-PRM-Qwen-2.5-1.5B     |  🤗[Skywork-o1-Open-PRM-Qwen-2.5-1.5B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen2.5-1.5B)|一个专门设计的模型，通过增量式过程奖励增强推理能力，非常适合小规模的复杂问题解决。|
| Skywork o1 Open-PRM-Qwen-2.5-7B     |  🤗[Skywork o1 Open-PRM-Qwen-2.5-7B](https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B)|在1.5B模型的基础上进一步扩展能力，能够处理更高要求的推理任务，推动人工智能推理的边界。|


## Huggingface 推理


1. 克隆skywork prm推理库
```shell
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
cd skywork-o1-prm-inference
```
2. 运行prm模型

```python
from transformers import AutoTokenizer
from model_utils.prm_model import PRM_MODEL
from model_utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards

prm_model_path = "/path/to/prm_model"
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is \\(\\boxed{18}\\)."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]


processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in datas]
input_ids, steps, reward_flags = zip(*processed_data)

model = PRM_MODEL.from_pretrained(prm_model_path, device_map="auto").eval()
input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(input_ids, reward_flags, tokenizer.pad_token_id)
_, _, rewards = model(input_ids=input_ids, attention_mask=attention_mask, return_probs=True)
step_rewards = derive_step_rewards(rewards, reward_flags)
print("step_rewards:",step_rewards[0])
print("step_rewards:",step_rewards[1])
```

## 通过vllm服务进行推理

1. 安装vllm并安装vllm-prm插件
```shell
pip install vllm==v0.6.4.post1
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git
cd skywork-o1-prm-inference
pip install -e .
```

2. 开启vllm服务
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /path/to/prm_model \
    --host 0.0.0.0 \
    --port 8081 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype auto
```

3. 发送推理请求

```python
from openai import OpenAI
from transformers import AutoTokenizer
from model_utils.io_utils import prepare_input, derive_step_rewards_vllm

prm_model_path = "/path/to/prm_model"
tokenizer = AutoTokenizer.from_pretrained(prm_model_path, trust_remote_code=True)
datas = [
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To determine how much money Janet makes every day at the farmers' market, we need to follow these steps:\n1. **Calculate the total number of eggs laid by the ducks per day.**\n   Janet's ducks lay 16 eggs per day.\n2. **Determine the number of eggs Janet uses each day.**\n   - She eats 3 eggs for breakfast every morning.\n   - She bakes muffins for her friends every day with 4 eggs.\n   So, the total number of eggs used per day is:\n   \\[\n   3 + 4 = 7 \\text{ eggs}\n   \\]\n3. **Calculate the number of eggs Janet sells at the farmers' market each day.**\n   Subtract the number of eggs used from the total number of eggs laid:\n   \\[\n   16 - 7 = 9 \\text{ eggs}\n   \\]\n4. **Determine how much money Janet makes from selling the eggs.**\n   She sells each egg for $2, so the total amount of money she makes is:\n   \\[\n   9 \\times 2 = 18 \\text{ dollars}\n   \\]\nTherefore, the amount of money Janet makes every day at the farmers' market is \\(\\boxed{18}\\)."
    },
    {
        "problem"  : "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        "response" : "To find out how much Janet makes every day at the farmers' market, we need to follow these steps:\n1. Calculate the total number of eggs laid by the ducks: \n   - Janet's ducks lay 16 eggs per day.\n2. Calculate the total number of eggs consumed by Janet and used for baking:\n   - Janet eats 3 eggs for breakfast.\n   - She bakes 4 eggs for muffins.\n   - Total eggs used: 3 (breakfast) + 4 (baking) = 7 eggs.\n3. Calculate the remaining number of eggs for sale:\n   - Total eggs laid: 16\n   - Eggs used: 7\n   - Remaining eggs: 16 - 7 = 9 eggs\n4. Calculate the total amount of money made at the farmers' market:\n   - Price per egg: $2\n   - Number of eggs sold: 9\n   - Total money made: 9 * $2 = $18\nTherefore, Janet makes $\\boxed{18}$ dollars every day at the farmers' market."
    }
]

# data preprocessing
processed_data = [prepare_input(d["problem"], d["response"], tokenizer=tokenizer, step_token="\n") for d in datas]
input_ids, steps, reward_flags = zip(*processed_data)

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8081/v1"
client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
models = client.models.list()
model = models.data[0].id
rewards = client.embeddings.create(
    input=input_ids,
    model=model,
)

step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
print("step_rewards:",step_rewards[0])
print("step_rewards:",step_rewards[1])  
```