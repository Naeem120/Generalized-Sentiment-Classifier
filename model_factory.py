from typing import List, Callable, Union
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

class ModelFactory:
    """A factory class to create models, tokenizers, and target modules for various pre-trained models."""
    
    def __init__(self, num_labels: int) -> None:
        """Initializes the ModelFactory with the specified number of labels."""
        self.num_labels: int = num_labels

    def get_model(self, model_name: str) -> PreTrainedModel:
        """Retrieves a pre-trained model for sequence classification based on the given model name."""
        model_registry: dict[str, Callable[[], PreTrainedModel]] = {
            'bert-base-uncased': lambda: AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased', num_labels=self.num_labels),
            'distilbert/distilbert-base-uncased': lambda: AutoModelForSequenceClassification.from_pretrained(
                'distilbert/distilbert-base-uncased', num_labels=self.num_labels),
            'distilbert/distilbert-base-uncased-finetuned-sst-2-english': lambda: AutoModelForSequenceClassification.from_pretrained(
                'distilbert/distilbert-base-uncased-finetuned-sst-2-english', num_labels=self.num_labels, ignore_mismatched_sizes=True),
            'roberta-base': lambda: AutoModelForSequenceClassification.from_pretrained(
                'roberta-base', num_labels=self.num_labels),
            'cardiffnlp/twitter-roberta-base-sentiment-latest': lambda: AutoModelForSequenceClassification.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment-latest', num_labels=self.num_labels, ignore_mismatched_sizes=True)
        }
        return model_registry.get(model_name, lambda: ValueError("Unknown model name"))()

    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Retrieves a tokenizer for the specified pre-trained model."""
        tokenizer_registry: dict[str, Callable[[], PreTrainedTokenizer]] = {
            'bert-base-uncased': lambda: AutoTokenizer.from_pretrained('bert-base-uncased'),
            'distilbert/distilbert-base-uncased': lambda: AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased'),
            'distilbert/distilbert-base-uncased-finetuned-sst-2-english': lambda: AutoTokenizer.from_pretrained(
                'distilbert/distilbert-base-uncased-finetuned-sst-2-english'),
            'roberta-base': lambda: AutoTokenizer.from_pretrained('roberta-base'),
            'cardiffnlp/twitter-roberta-base-sentiment-latest': lambda: AutoTokenizer.from_pretrained(
                'cardiffnlp/twitter-roberta-base-sentiment-latest')
        }
        return tokenizer_registry.get(model_name, lambda: ValueError("Unknown tokenizer name"))()

    def get_target_modules(self, model_name: str) -> List[str]:
        """Retrieves a list of target modules for the specified model."""
        target_modules_registry: dict[str, List[str]] = {
            'bert-base-uncased': ['query', 'key', 'value'],
            'distilbert/distilbert-base-uncased': ['q_lin', 'k_lin', 'v_lin'],
            'distilbert/distilbert-base-uncased-finetuned-sst-2-english': ['q_lin', 'k_lin', 'v_lin'],
            'roberta-base': ['query', 'key', 'value'],
            'cardiffnlp/twitter-roberta-base-sentiment-latest': ['query', 'key', 'value']
        }
        return target_modules_registry[model_name]
