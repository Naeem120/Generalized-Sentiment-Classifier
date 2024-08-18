import torch
from model_pipeline_manager import ModelPipelineManager
from pytorch_dataset_converter import TextClassificationDataset
from trainers import StandardTrainer, LoRATrainer


class SentimentAnalysisTool:
    """
    A tool for sentiment analysis that supports both standard and LoRA-based fine-tuning.
    
    Attributes:
        use_lora (bool): Whether to use LoRA for fine-tuning.
        instantiate_model_pipeline (ModelPipelineManager): Manages model and tokenizer loading.
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """

    def __init__(self, model_name: str = None, num_labels: int = 2, use_lora: bool = False, device: str = 'cuda'):
        """Initializes the SentimentAnalysisTool with the specified model, labels, and configuration.
        
        Args:
            model_name (str, optional): The name of the model to load. Defaults to None.
            num_labels (int, optional): The number of labels for classification. Defaults to 2.
            use_lora (bool, optional): Whether to use LoRA for fine-tuning. Defaults to False.
            device (str, optional): The device to run the model on. Defaults to 'cuda'.
        """
        self.use_lora = use_lora
        self.instantiate_model_pipeline = ModelPipelineManager(model_name, num_labels, use_lora)
        self.model, self.tokenizer = self.instantiate_model_pipeline.load_model_and_tokenizer()
        self.model, self.tokenizer = ModelPipelineManager(model_name, num_labels, use_lora).load_model_and_tokenizer()
        self.device = torch.device(device)
        self.model.to(self.device)

    def preprocess_data(self, train_texts: list[str], train_labels: list[int], eval_texts: list[str], eval_labels: list[int]) -> tuple[TextClassificationDataset, TextClassificationDataset]:
        """Preprocesses the training and evaluation data using the tokenizer.
        
        Args:
            train_texts (list[str]): The texts for training.
            train_labels (list[int]): The labels for training.
            eval_texts (list[str]): The texts for evaluation.
            eval_labels (list[int]): The labels for evaluation.
        
        Returns:
            tuple[TextClassificationDataset, TextClassificationDataset]: The preprocessed training and evaluation datasets.
        """
        self.preprocessed_train_data = TextClassificationDataset(train_texts, train_labels, self.tokenizer)
        self.preprocessed_eval_data = TextClassificationDataset(eval_texts, eval_labels, self.tokenizer)
        return self.preprocessed_train_data, self.preprocessed_eval_data
    
    def fine_tune(self):
        """Fine-tunes the model using either a standard trainer or a LoRA trainer."""
        if self.use_lora:
            self.lora_trainer = LoRATrainer(self.model, self.tokenizer, self.preprocessed_train_data, self.preprocessed_eval_data)
            self.lora_trainer.train()
        else:
            self.standard_trainer = StandardTrainer(self.model, self.tokenizer, self.preprocessed_train_data, self.preprocessed_eval_data)
            self.standard_trainer.train()
