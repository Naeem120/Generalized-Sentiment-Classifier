import time
from typing import Optional
import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR



class StandardTrainer:
    """
    A class to manage the training and evaluation of a model using Hugging Face's Trainer API.
    
    Attributes:
        model: The model to be trained.
        train_dataset: The dataset used for training.
        eval_dataset: The dataset used for evaluation.
    """
    
    def __init__(self, model, train_dataset, eval_dataset):
        """Initializes the StandardTrainer with the model and datasets.
        
        Args:
            model: The model to be trained.
            train_dataset: The dataset used for training.
            eval_dataset: The dataset used for evaluation.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def train(self, 
              batch_size: int = 32, 
              epochs: int = 3, 
              evaluation_strategy: str = 'epoch', 
              save_strategy: str = 'epoch', 
              logging_steps: int = 10,
              load_best_model_at_end: bool = True,
              learning_rate: Optional[float] = 5e-5,
              **kwargs):
        """Trains the model using the specified batch size, number of epochs, and other customizable settings.
        
        Args:
            batch_size (int, optional): The batch size for training. Defaults to 32.
            epochs (int, optional): The number of training epochs. Defaults to 3.
            evaluation_strategy (str, optional): The strategy for evaluation during training. Defaults to 'epoch'.
            save_strategy (str, optional): The strategy for saving the model during training. Defaults to 'epoch'.
            logging_steps (int, optional): The frequency of logging training metrics. Defaults to 10.
            load_best_model_at_end (bool, optional): Whether to load the best model at the end of training. Defaults to True.
            learning_rate (float, optional): The learning rate for training. Optional. Defaults to the model's default.
            **kwargs: Additional keyword arguments for the `TrainingArguments` class.
        """
        
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy=evaluation_strategy,
            save_strategy=save_strategy,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            logging_dir='./logs',
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            learning_rate=learning_rate,  # Optional: Use model's default if not provided
            **kwargs  # Allows additional customization through TrainingArguments
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )

        trainer.train()


class LoRATrainer:
    """
    A trainer class for fine-tuning a model with LoRA using PyTorch.
    
    Attributes:
        model: The model to be trained.
        tokenizer: The tokenizer associated with the model.
        train_dataloader: DataLoader for the training dataset.
        eval_dataloader: DataLoader for the evaluation dataset.
        optimizer: Optimizer for training.
        lr_scheduler: Learning rate scheduler.
        device: The device to run the training on (CPU or GPU).
    """

    def __init__(self, model, tokenizer, train_dataset, eval_dataset, batch_size: int = 32, optimizer_cls=AdamW, learning_rate: float = 5e-5, epochs: int = 3):
        """Initializes the LoRATrainer with model, tokenizer, datasets, and training parameters.
        
        Args:
            model: The model to be trained.
            tokenizer: The tokenizer associated with the model.
            train_dataset: The dataset used for training.
            eval_dataset: The dataset used for evaluation.
            batch_size (int, optional): The batch size for DataLoader. Defaults to 32.
            optimizer_cls (type, optional): The optimizer class to use. Defaults to AdamW.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 5e-5.
            epochs (int, optional): The number of training epochs. Defaults to 3.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)
        self.optimizer = optimizer_cls(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.95)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.epochs = epochs

    def train(self):
        """Trains the model for the specified number of epochs."""
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0
            epoch_start_time = time.time()
            
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(self.train_dataloader)
            epoch_end_time = time.time()
            print(f"Epoch {epoch + 1}/{self.epochs} - Training Loss: {avg_train_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f}s")
            self.evaluate()

    def evaluate(self):
        """Evaluates the model on the evaluation dataset."""
        self.model.eval()
        eval_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                eval_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch["labels"]).sum().item()
                total_predictions += batch["labels"].size(0)
        
        avg_eval_loss = eval_loss / len(self.eval_dataloader)
        accuracy = correct_predictions / total_predictions
        print(f"Validation Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}")