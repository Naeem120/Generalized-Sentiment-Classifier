from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import Dataset

class TextClassificationDataset(Dataset):
    """A PyTorch Dataset for text classification tasks."""
    
    def __init__(self, texts: list[str], labels: list[int], tokenizer: PreTrainedTokenizer, max_length: int = 128):
        """Initializes the dataset with texts, labels, and a tokenizer.
        
        Args:
            texts (list[str]): The list of input texts.
            labels (list[int]): The list of corresponding labels.
            tokenizer (PreTrainedTokenizer): The tokenizer used to process the texts.
            max_length (int, optional): The maximum length for tokenized inputs. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = self.tokenizer(self.texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')

    def __len__(self) -> int:
        """Returns the number of examples in the dataset."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves the encoded inputs and label for a given index.
        
        Args:
            idx (int): The index of the item to retrieve.
        
        Returns:
            dict[str, torch.Tensor]: A dictionary containing tokenized inputs and the label.
        """
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item