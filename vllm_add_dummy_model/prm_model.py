import os
import torch
import re
import sys
import vllm
from torch import nn
from vllm.model_executor.layers.pooler import (
    Optional,
    List,
    PoolerConfig,
    PoolingType,
    PoolingMetadata,
    PoolingTensors,
    EmbeddingSequenceGroupOutput,
    PoolerOutput
)
from vllm.model_executor.models.qwen2_rm import (
    Qwen2Model,
    AutoWeightsLoader,
    VllmConfig,
    maybe_prefix,
    IntermediateTensors,
    PoolingMetadata,
    PoolingType,
    PoolerOutput,
    AttentionMetadata,
    Iterable,
    Union,
    Tuple,
    SupportsPP
)

class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        # print('enter here')
        # print('output1.shape: ', output.shape)
        output = self.summary(output)
        # print('output2.shape: ', output.shape)
        return output

    


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use.
        normalize: Whether to normalize the pooled data.
    """

    def __init__(
        self,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ):
        super().__init__()

        self.pooling_type = pooling_type
        self.normalize = normalize
        self.softmax = softmax
        self.step_tag_id = step_tag_id
        self.returned_token_ids = returned_token_ids

    @classmethod
    def from_config_with_defaults(
        cls,
        pooler_config: PoolerConfig,
        pooling_type: PoolingType,
        normalize: bool,
        softmax: bool,
        step_tag_id: Optional[int] = None,
        returned_token_ids: Optional[List[int]] = None,
    ) -> Optional["Pooler"]:
        if pooler_config is None:
            return None
        return cls(
            pooling_type=PoolingType[pooler_config.pooling_type]
            if pooler_config.pooling_type is not None else pooling_type,
            normalize=pooler_config.normalize
            if pooler_config.normalize is not None else normalize,
            softmax=pooler_config.softmax
            if pooler_config.softmax is not None else softmax,
            step_tag_id=pooler_config.step_tag_id
            if pooler_config.step_tag_id is not None else step_tag_id,
            returned_token_ids=pooler_config.returned_token_ids
            if pooler_config.returned_token_ids is not None else
            returned_token_ids,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools specific information from hidden states based on metadata."""

        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        if self.pooling_type is PoolingType.CLS:
            first_token_flat_indices = torch.zeros_like(prompt_lens)
            first_token_flat_indices[1:] += torch.cumsum(prompt_lens,
                                                        dim=0)[:-1]
            pooled_data = hidden_states[first_token_flat_indices]
        elif self.pooling_type == PoolingType.LAST:
            last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_flat_indices]
        elif self.pooling_type == PoolingType.ALL:
            offset = 0
            pooled_data = []
            for prompt_len in prompt_lens:
                pooled_data.append(hidden_states[offset:offset + prompt_len])
                offset += prompt_len
            # pooled_data = torch.stack(pooled_data_lst)
        elif self.pooling_type == PoolingType.MEAN:
            # Calculate mean pooling
            cumsum = torch.cumsum(hidden_states, dim=0)
            start_indices = torch.cat([
                torch.tensor([0], device=hidden_states.device),
                torch.cumsum(prompt_lens[:-1], dim=0)
            ])
            end_indices = torch.cumsum(prompt_lens, dim=0)
            pooled_data = (
                cumsum[end_indices - 1] - cumsum[start_indices] +
                hidden_states[start_indices]) / prompt_lens.unsqueeze(1)
        elif self.pooling_type == PoolingType.STEP:
            returned_token_ids = self.returned_token_ids
            if returned_token_ids is not None and len(returned_token_ids) > 0:
                hidden_states = hidden_states[:, returned_token_ids]

            step_tag_id = self.step_tag_id

            offset = 0
            pooled_data_lst = []
            for prompt_len, seq_data_i in zip(
                    prompt_lens, pooling_metadata.seq_data.values()):
                pooled_data_i = hidden_states[offset:offset + prompt_len]
                if step_tag_id is not None:
                    token_ids = torch.tensor(seq_data_i.prompt_token_ids)
                    pooled_data_i = pooled_data_i[token_ids == step_tag_id]

                offset += prompt_len
                pooled_data_lst.append(pooled_data_i)

            pooled_data = torch.stack(pooled_data_lst)
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        if self.softmax:
            pooled_data = nn.functional.softmax(pooled_data, dim=-1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)

class Qwen2ForPrmModel(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        pooler_config = vllm_config.model_config.pooler_config
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                            "supported. This model uses sliding window "
                            "but `max_window_layers` = {} is less than "
                            "`num_hidden_layers` = {}. Please open an issue "
                            "to discuss this feature.".format(
                                config.max_window_layers,
                                config.num_hidden_layers,
                            ))

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.v_head = ValueHead(self.config)

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.ALL,
            normalize=False,
            softmax=False)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                attn_metadata, intermediate_tensors)
        logits = self.v_head(hidden_states)
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self,
                                ignore_unexpected_prefixes=["lm_head."])
        loader.load_weights(weights)

def register():
    from vllm import ModelRegistry
    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2ForPrmModel", "vllm_add_dummy_model.prm_model:Qwen2ForPrmModel")
