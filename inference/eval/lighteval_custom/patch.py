from __future__ import annotations
import lighteval.models.vllm.vllm_model as vllm_model

from typing import Optional

from pydantic import NonNegativeFloat, NonNegativeInt, PositiveInt
from tqdm import tqdm

from lighteval.data import GenerativeTaskDataset
from lighteval.models.model_output import GenerativeResponse

from lighteval.models.utils import ModelConfig
from lighteval.tasks.requests import GreedyUntilRequest

from lighteval.utils.utils import as_list
from vllm import LLM

import ray
from more_itertools import distribute
from vllm import LLM, SamplingParams
import itertools


class VLLMModelConfigEdit(ModelConfig):
    model_name: str
    revision: str = "main"  # revision of the model
    dtype: str = "bfloat16"
    tensor_parallel_size: PositiveInt = 1  # how many GPUs to use for tensor parallelism
    data_parallel_size: PositiveInt = 1  # how many GPUs to use for data parallelism
    pipeline_parallel_size: PositiveInt = (
        1  # how many GPUs to use for pipeline parallelism
    )
    gpu_memory_utilization: NonNegativeFloat = (
        0.9  # lower this if you are running out of memory
    )
    max_model_length: PositiveInt | None = (
        None  # maximum length of the model, ussually infered automatically. reduce this if you encouter OOM issues, 4096 is usually enough
    )
    swap_space: PositiveInt = 4  # CPU swap space size (GiB) per GPU.
    seed: NonNegativeInt = 1234
    trust_remote_code: bool = False
    use_chat_template: bool = False
    add_special_tokens: bool = True
    multichoice_continuations_start_space: bool = (
        True  # whether to add a space at the start of each continuation in multichoice generation
    )
    pairwise_tokenization: bool = (
        False  # whether to tokenize the context and continuation separately or together.
    )
    max_num_seqs: PositiveInt = (
        128  # maximum number of sequences per iteration; This variable and `max_num_batched_tokens` effectively control the batch size at prefill stage. See https://github.com/vllm-project/vllm/issues/2492 for detailed explaination.
    )
    max_num_batched_tokens: PositiveInt = 2048  # maximum number of tokens per batch
    subfolder: str | None = None
    enforce_eager: bool = True


vllm_model.VLLMModelConfig = VLLMModelConfigEdit


def _create_auto_model(self, config: vllm_model.VLLMModelConfig) -> Optional[LLM]:
        """
        Creates an instance of the pretrained HF model.

        Args:
            pretrained (str): The name or path of the pretrained model.
            revision (str): The revision of the model.
            subfolder (Optional[str], optional): The subfolder within the model. Defaults to None.
            max_memory (Optional[dict], optional): The maximum memory to allocate for the model per GPU. Defaults to None.
            device_map (Optional[dict], optional): The device mapping for the model. Defaults to None.
            torch_dtype (Optional[Union[str, torch.dtype]], optional): The torch data type for the model. Defaults to None.
            quantization_config (Optional[Union[BitsAndBytesConfig, GPTQConfig]], optional): The quantization configuration for the model. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code. Defaults to False.
            cache_dir (str, optional): The cache directory for the model. Defaults to "/scratch".

        Returns:
            transformers.PreTrainedModel: The created auto model instance.
        """
        self.model_args = {
            "model": config.model_name,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "revision": config.revision + (f"/{config.subfolder}" if config.subfolder is not None else ""),
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "max_model_len": self._max_length,
            "swap_space": 4,
            "seed": int(config.seed),
            "max_num_seqs": int(config.max_num_seqs),
            "max_num_batched_tokens": int(config.max_num_batched_tokens),
            "enforce_eager": config.enforce_eager,
        }

    

        if config.data_parallel_size > 1:
            self.model_args["distributed_executor_backend"] = "ray"
            self._batch_size = "auto"
            return None
    

        model = LLM(**self.model_args)

        # If the max_length can't get extracted from the config, it will be inferred from the model
        # Inferring from the tokenizer will cause vllm to bug for models with mismatches between model
        # config and tk config, like mistralai/Mistral-7B-v0.1
        if self._max_length is None:
            self._max_length = model.llm_engine.model_config.max_seq_len_to_capture

        return model


def greedy_until(
    self,
    requests: list[GreedyUntilRequest],
    override_bs: Optional[int] = None,
) -> list[GenerativeResponse]:
    """
    Generates responses using a greedy decoding strategy until certain ending conditions are met.
    Directly calls vLLM with string prompts.
    """
    for request in requests:
        request.stop_sequence = as_list(request.stop_sequence) + [
            self.tokenizer.eos_token
        ]
        # 使用 lighteval 模型包装器提供的 tok_encode 方法
        # (假设 VLLMModel 继承了 LightevalModel 并有 self.tok_encode)
        request.tokenized_context = self.tok_encode(request.context)

    dataset = GenerativeTaskDataset(
        requests=requests, num_dataset_splits=self.DATASET_SPLITS
    )
    results = []

    for split in tqdm(
        dataset.splits_iterator(),
        total=dataset.num_dataset_splits,
        desc="Splits",
        position=0,
        disable=False,
    ):
        # --- Prepare parameters for direct vLLM call ---

        # 1. Get string contexts for the batch
        contexts: list[str] = [sample.context for sample in split]

        # 2. Determine stop tokens (handle chat template case)
        if self.use_chat_template:
            stop_tokens = []
        else:
            # Assume same stop sequence for the batch (as before)
            stop_tokens = split[0].stop_sequence

        # 3. Determine max_new_tokens
        # Use generation_size from the request if available, else default
        max_new_tokens = (
            self._config.generation_parameters.max_new_tokens
            or split[0].generation_size
        )

        # 4. Check if logits are needed (for post-processing)
        returns_logits = split[0].use_logits

        # 5. Build SamplingParams (for greedy)
        # We build this here instead of in _generate
        sampling_params = SamplingParams(
            **self._config.generation_parameters.to_vllm_dict()  # Start with defaults
        )
        # Override for greedy and specific request needs
        sampling_params.temperature = 0.0
        sampling_params.top_p = 1.0
        sampling_params.top_k = -1
        sampling_params.n = split[0].num_samples  # Number of samples per prompt
        sampling_params.max_tokens = max_new_tokens
        sampling_params.stop = stop_tokens
        sampling_params.logprobs = (
            1 if returns_logits else None
        )  # Request logprobs if needed

        # --- ★ Direct Call to vLLM Generate ★ ---
        # We bypass _generate entirely
        # NOTE: Truncation is now handled by vLLM based on max_model_len set during LLM init.
        # Ensure max_model_len is appropriate. The previous warning suggests it might be too low.
        if self.data_parallel_size > 1:
            # --- Handle Ray distribution if needed ---
            @ray.remote(num_gpus=1 if self.tensor_parallel_size == 1 else None)
            def run_inference_one_model(
                model_args: dict, sampling_params: SamplingParams, requests: list[str]
            ):
                llm = LLM(**model_args)
                return llm.generate(prompts=requests, sampling_params=sampling_params)

            requests_ray = [
                list(x) for x in distribute(self.data_parallel_size, contexts)
            ]
            inputs_ray = (
                (self.model_args, sampling_params, req) for req in requests_ray
            )
            object_refs = [run_inference_one_model.remote(*x) for x in inputs_ray]
            results_ray = ray.get(object_refs)
            ray.shutdown()
            # Flatten results
            vllm_outputs = [
                x
                for x in itertools.chain.from_iterable(
                    itertools.zip_longest(*[list(x) for x in results_ray])
                )
                if x is not None
            ]
        else:
            # --- Single process call ---
            vllm_outputs = self.model.generate(
                prompts=contexts,  # Pass strings directly
                sampling_params=sampling_params,
                use_tqdm=True,  # You can keep this
            )
        # --- End Direct Call ---

        # --- Post-processing (remains largely the same) ---
        # This loop now iterates through the direct output of self.model.generate
        for i, vllm_output in enumerate(vllm_outputs):
            # Ensure index aligns with original split sample if needed
            # original_sample = split[i]

            output_token_ids = [outputs.token_ids for outputs in vllm_output.outputs]
            logprobs_list = (
                [output.logprobs for output in vllm_output.outputs]
                if returns_logits and vllm_output.outputs[0].logprobs
                else []
            )

            final_logprobs = []
            if logprobs_list:
                # (Keep the logprob processing logic from your previous code)
                current_logprobs = logprobs_list[0]
                current_tokens = output_token_ids[0]
                if len(current_logprobs) == len(current_tokens):
                    final_logprobs = [
                        current_logprobs[i].get(current_tokens[i])
                        for i in range(len(current_tokens))
                    ]
                    final_logprobs = [
                        lp.logprob if lp is not None else -float("inf")
                        for lp in final_logprobs
                    ]
                else:
                    print(
                        f"Warning: Mismatch between logprobs ({len(current_logprobs)}) and tokens ({len(current_tokens)}). Logprobs skipped."
                    )

            result_texts = [output.text for output in vllm_output.outputs]
            input_token_ids = vllm_output.prompt_token_ids

            cur_response = GenerativeResponse(
                result=result_texts,
                logits=final_logprobs,
                generated_tokens=list(output_token_ids),  # Ensure list[list[int]]
                input_tokens=input_token_ids,
            )
            results.append(cur_response)

        return dataset.get_original_order(results)


vllm_model.VLLMModel.greedy_until = greedy_until
vllm_model.VLLMModel._create_auto_model = _create_auto_model
