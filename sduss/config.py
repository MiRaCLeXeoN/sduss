"""All configuration classes

Including ModelConfig, 
"""

class ModelConfig:
    """Configuration for model.
    
    Models are default to use from huggingface models.
    
    Attributes:
        model: Name or path of the huggingface model to use.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: 
    
    """
    
    def __init__(
        self,
        model: str,
        tokenizer: str,
        tokenizer_mode: str,
    ) -> None:
        """init method
        
        
        """
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        
    def _verify_tokenizer_mode(self) -> None:
        """Verify that tokenizer_mode is either 'auto' or 'slow'."""
        tokenizer_mode = self.tokenizer_mode.lower()
        # ? what are these modes? what's the difference?
        if tokenizer_mode not in ["auto", "slow"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be"
                "either 'auto' or 'slow'."
            )
        self.tokenizer_mode = tokenizer_mode
        