import os 
from peft import LoraConfig, get_peft_model
from model_factory import ModelFactory

class ModelPipelineManager:
    """
    Manages the loading, saving, and configuration of models and tokenizers, including optional LoRA support.
    
    Attributes:
        model_name (str): The name of the model to load.
        num_labels (int): The number of labels for the model.
        use_lora (bool): Whether to use LoRA for the model.
    """

    def __init__(self, model_name: str = None, num_labels: int = 2, use_lora: bool = False):
        """Initializes the ModelPipelineManager with the given model name, number of labels, and LoRA usage.
        
        Args:
            model_name (str, optional): The name of the model to load. Defaults to None.
            num_labels (int, optional): The number of labels for classification. Defaults to 2.
            use_lora (bool, optional): Whether to use LoRA for fine-tuning. Defaults to False.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_lora = use_lora 
        self.target_modules = None
        self.factory = ModelFactory(self.num_labels)

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Loads the model and tokenizer based on the provided model name, with optional LoRA configuration.
        
        Returns:
            tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer.
        
        Raises:
            ValueError: If no model name is provided.
        """
        if not self.model_name:
            raise ValueError("No model name provided. Please provide a model name or load a saved model.")
        
        self.model = self.factory.get_model(self.model_name)
        self.tokenizer = self.factory.get_tokenizer(self.model_name)
        
        if self.use_lora:
            self.target_modules = self.factory.get_target_modules(self.model_name)
            lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=self.target_modules)
            self.model = get_peft_model(self.model, lora_config)
        
        return self.model, self.tokenizer

    def save_pipeline(self, output_dir: str):
        """Saves the model and tokenizer to the specified directory.
        
        Args:
            output_dir (str): The directory where the model and tokenizer will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def load_pipeline(self, output_dir: str):
        """Loads the model and tokenizer from the specified directory.
        
        Args:
            output_dir (str): The directory where the model and tokenizer are saved.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(output_dir)
