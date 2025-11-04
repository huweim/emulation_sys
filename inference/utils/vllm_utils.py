# inference/utils/vllm_utils.py
import tempfile
import os
import glob
import shutil
import logging
from transformers import AutoConfig

logger = logging.getLogger(__name__)

def prepare_vllm_temp_model(original_model_path: str, 
                           architecture_override: str, 
                           quant_params: dict = None) -> str:
    """
    Create or update a persistent cache model directory with a modified config for vLLM.

    Logic:
    1. Check if the cache directory exists.
    2. If it doesn't exist, create it, symlink the weights, and copy all non-weight files.
    3. Regardless of whether the directory exists, always reload the original config,
       apply 'architecture_override' and 'quant_params',
       and overwrite 'config.json' in the cache directory.
    """
    
    # 1. Create a deterministic directory name
    base_model_name = os.path.basename(original_model_path.rstrip('/'))
    safe_override_name = architecture_override.replace('/', '_')
    cache_dir_name = f"vllm-cache-{base_model_name}-{safe_override_name}"
    cache_path = os.path.join(tempfile.gettempdir(), cache_dir_name)

    # 2. Check if the directory exists
    if not os.path.exists(cache_path):
        logger.info(f"[VLLM_Util] Cache dir not found. Creating at: {cache_path}")
        try:
            os.makedirs(cache_path, exist_ok=True)

            # 2a. Symlink the checkpoint files
            weight_files = glob.glob(os.path.join(original_model_path, "*.safensors"))
            weight_files += glob.glob(os.path.join(original_model_path, "*.bin"))
            for f in weight_files:
                os.symlink(f, os.path.join(cache_path, os.path.basename(f)))
            
            # 2b. Copy all other config and tokenizer files
            other_files_patterns = ["*.json", "*.model", "*.py", "tokenizer*"]
            for pattern in other_files_patterns:
                for f in glob.glob(os.path.join(original_model_path, pattern)):
                    # Don't copy config.json, as we will generate it
                    if os.path.basename(f) == "config.json":
                        continue 
                    dest_path = os.path.join(cache_path, os.path.basename(f))
                    if os.path.isfile(f):
                        shutil.copy(f, dest_path)
                    elif os.path.isdir(f) and not os.path.exists(dest_path):
                        shutil.copytree(f, dest_path)
            logger.info(f"[VLLM_Util] Cache dir structure created.")
        except Exception as e:
            logger.error(f"[VLLM_Util] Error creating cache vLLM directory: {e}")
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path) # Clean up if creation fails
            raise e
    else:
         logger.info(f"[VLLM_Util] Cache dir found. Reusing structure at: {cache_path}")

    # 3. Regenerate and overwrite config.json
    try:
        logger.info(f"[VLLM_Util] Overwriting config.json in cache with current args...")
        config = AutoConfig.from_pretrained(original_model_path) # Always load from the original path
        
        # 3a. Set architecture
        config.architectures = [architecture_override]
        
        # 3b. Write quantization parameters to config.json
        if quant_params is not None:
            config.quantization_config = quant_params

        # 3c. Save the modified config to the cache directory
        config.save_pretrained(cache_path)
        
    except Exception as e:
        logger.error(f"[VLLM_Util] Error overwriting config.json: {e}")
        raise e

    return cache_path