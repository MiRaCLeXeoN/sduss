from typing import Optional

from transformers import AutoConfig, PretrainedConfig

# Here lists all supported configurations, which are inherited from
# transformers.configuration_utils.PretrainedConfig
_CONFIG_REGISTRY = {
    # ! incomplete
}

def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
) -> PretrainedConfig:
    try:
        config = AutoConfig.from_pretrained(model, 
            trust_remote_code=trust_remote_code, revision=revision)
    except ValueError as e:
        if (not trust_remote_code and
            "requires you to execute the configuration file" in str(e)):
            # intercepts if the error is raised due to trust_remote_code setting
            err_msg = (
                "Failed to load the model config. If the model is a custom "
                "model not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` "
                "or using the `--trust-remote-code` flag in the CLI.")
            raise RuntimeError(err_msg) from e
        else:
            raise e
    
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model, revision=revision)
    
    return config