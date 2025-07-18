# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

class EmbeddingModel(nn.Module):
    def __init__(self, model_name: str = "Salesforce/SFR-Embedding-Mistral", torch_dtype=torch.float16, device_map=None):
        """
        Initialize the embedding model.

        Args:
            model_name (str): The name of the pre-trained model from Hugging Face.
        """
        super().__init__()
        # Set the model name based on the short name
        if model_name == "sfr":
            model_name = "Salesforce/SFR-Embedding-Mistral"
        elif model_name == "simcse":
            model_name = "princeton-nlp/sup-simcse-roberta-large"
        elif model_name == "bert":
            model_name = "bert-base-uncased"
        elif model_name == "e5":
            model_name = 'intfloat/e5-large'

        self.model_name = model_name
        self.model = AutoModel.from_pretrained(
            model_name, 
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer

    @staticmethod
    def last_token_pool(
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Implemented at https://huggingface.co/Salesforce/SFR-Embedding-Mistral"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        
    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
    def tokenize(self, texts, max_length: int = 1024):
        """
        Tokenize the input texts.

        Args:
            texts (list of str): The input texts to tokenize.
            max_length (int): The maximum sequence length for tokenization.

        Returns:
            dict: The tokenized inputs.
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def forward(self, **inputs):
        if self.model_name == "Salesforce/SFR-Embedding-Mistral":
            outputs = self.model(**inputs)
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        elif self.model_name == "princeton-nlp/sup-simcse-roberta-large":
            embeddings = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        elif self.model_name == "bert-base-uncased":
            outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden_state = outputs.last_hidden_state
            embeddings = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        elif self.model_name == "intfloat/e5-large":
            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        else:
            raise ValueError(f"Model {self.model_name} not supported.")
    
        return F.normalize(embeddings, p=2, dim=1)
        
    def encode(self, texts, max_length: int = 1024, batch_size: int = 64, show_progress_bar=True) -> torch.Tensor:
        """
        Generate embeddings for a list of input texts, with batching if input size exceeds batch_size.

        Args:
            texts (list of str): The input texts to encode.
            max_length (int): The maximum sequence length for tokenization.
            batch_size (int): The batch size for processing inputs.
            show_progress_bar (bool): Whether to show a progress bar during processing.

        Returns:
            torch.Tensor: The embeddings for the input texts.
        """
        all_embeddings = []

        # assert the texts is a list of strings
        assert isinstance(texts, list) and all(isinstance(text, str) for text in texts), "Input must be a list of strings: {}".format(texts)

        index_iterator = range(0, len(texts), batch_size)
        index_iterator = tqdm(index_iterator, desc="Processing batches") if show_progress_bar else index_iterator
        for i in index_iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize the input texts
            inputs = self.tokenize(batch_texts, max_length=max_length).to(self.model.device)

            # Forward pass through the model
            with torch.no_grad():
                embeddings = self(**inputs)

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)

# Example usage
if __name__ == "__main__":
    model = EmbeddingModel(model_name="princeton-nlp/sup-simcse-roberta-large")
    texts = ["This is a test sentence.", "Another example sentence."]
    embeddings = model.encode(texts)
    print(embeddings.shape)