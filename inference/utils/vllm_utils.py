# inference/utils/vllm_utils.py
import tempfile
import os
import glob
import shutil
import logging
from transformers import AutoConfig

logger = logging.getLogger(__name__)

def prepare_vllm_temp_model(original_model_path: str, architecture_override: str) -> str:
    """
    为 vLLM 创建一个持久化的、带有修改后 config 的缓存模型目录。

    这个函数会：
    1. 根据模型路径和重写名称，生成一个确定性的缓存目录路径（例如 /tmp/vllm-cache-Llama-2-7b-hf-Qwen2ForCausalLM_nvfp）。
    2. 检查该目录是否已存在。
    3. 如果存在，立即返回路径，跳过所有文件操作。
    4. 如果不存在，则创建目录、软链接权重、复制配置文件，并保存修改后的 config.json。

    Args:
        original_model_path (str): 原始模型（如 Llama-2-7b）的路径。
        architecture_override (str): 要写入新 config 的 'architectures' 名称 (例如 "Qwen2ForCausalLM_nvfp")。

    Returns:
        str: 缓存目录的路径。
    
    Raises:
        Exception: 如果在 *首次创建* 过程中发生任何文件操作失败。
    """
    
    # 1. 创建一个确定性的目录名称
    base_model_name = os.path.basename(original_model_path.rstrip('/'))
    # 替换斜杠等非法字符，以防 architecture_override 包含它们
    safe_override_name = architecture_override.replace('/', '_')
    cache_dir_name = f"vllm-cache-{base_model_name}-{safe_override_name}"
    
    # 将缓存目录放在系统的标准临时文件夹中 (例如 /tmp)
    cache_path = os.path.join(tempfile.gettempdir(), cache_dir_name)

    # 2. 检查目录是否已存在
    if os.path.exists(cache_path):
        logger.info(f"[VLLM_Util] Cache dir already exists. Reusing: {cache_path}")
        return cache_path

    # 3. 如果不存在，创建它
    logger.info(f"[VLLM_Util] Creating persistent cache dir at: {cache_path}")
    
    try:
        os.makedirs(cache_path, exist_ok=True) # 创建目录

        # 3a. 软链接 (Symlink) 巨大的权重文件
        weight_files = glob.glob(os.path.join(original_model_path, "*.safetensors"))
        weight_files += glob.glob(os.path.join(original_model_path, "*.bin"))
        for f in weight_files:
            os.symlink(f, os.path.join(cache_path, os.path.basename(f)))
        
        # 3b. 复制 (Copy) 所有其他配置文件和 tokenizer
        other_files_patterns = ["*.json", "*.model", "*.py", "tokenizer*"]
        for pattern in other_files_patterns:
            for f in glob.glob(os.path.join(original_model_path, pattern)):
                dest_path = os.path.join(cache_path, os.path.basename(f))
                if os.path.isfile(f):
                    shutil.copy(f, dest_path)
                elif os.path.isdir(f) and not os.path.exists(dest_path):
                    shutil.copytree(f, dest_path)

        # 3c. 修改临时目录中的 config.json
        config = AutoConfig.from_pretrained(original_model_path) # 从原始路径加载
        config.architectures = [architecture_override]
        config.save_pretrained(cache_path) # 保存到新路径

        logger.info(f"[VLLM_Util] Cache dir created successfully: {cache_path}")
        return cache_path

    except Exception as e:
        # 如果创建过程中失败，立即清理
        logger.error(f"[VLLM_Util] Error creating cache vLLM directory: {e}")
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
        raise e # 重新抛出异常，让主程序知道失败了