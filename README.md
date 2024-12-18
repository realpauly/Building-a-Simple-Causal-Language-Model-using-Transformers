# Building-a-Simple-Causal-Language-Model-using-Transformers
## **Introduction**

In this project, we aim to build a simple causal language model using **PyTorch** and the **Transformers library**. The model will be trained on the **WikiQA dataset**, a small and older **Question-Answer** dataset, available [here](https://www.microsoft.com/en-us/download/details.aspx?id=52419).  

The primary objective is to develop a **causal model** that can generate text by predicting the next word in a sequence, based on previously seen tokens. Unlike sequence-to-sequence models, causal models focus on autoregressive generation, where each token depends only on its preceding tokens.  

# **Code Explanation for Building a Causal Language Model**

1. **Data Preprocessing**:  
   - The dataset is cleaned to remove unnecessary characters and tokenized using a pre-trained tokenizer (e.g., GPT-2 tokenizer).  
   - The tokenized text is split into training and validation sets, and sequences are padded or truncated to a fixed length.

2. **Model Architecture**:  
   - A **transformer-based causal model** is implemented. The architecture includes an embedding layer, transformer encoder layers, and a linear output layer.  
   - A **causal mask** ensures autoregressive behavior by preventing the model from attending to future tokens during training.

3. **Training and Validation**:  
   - The model is trained using **cross-entropy loss** to predict the next token in a sequence.  
   - The training loop minimizes the loss through backpropagation, while the validation loop evaluates the model's performance on unseen data.

4. **Text Generation**:  
   - The trained model generates text autoregressively. Starting with a user-provided **prompt**, the model predicts one token at a time and appends it to the sequence until reaching the desired length.

5. **Attention Weight Visualization**:  
   - Attention weights from the transformer are extracted and visualized as **heatmaps**. These weights illustrate which parts of the input sequence the model attends to while making predictions.


## **Results**  
The output includes:  
### Validation loss, showing the model's performance.

  val_data = torch.load(VAL_FILE)
Starting training...
Epoch [1/5], Train Loss: 7.8431, Val Loss: 7.2523
Checkpoint saved!
Epoch [2/5], Train Loss: 6.7707, Val Loss: 6.6990
Checkpoint saved!
Epoch [3/5], Train Loss: 6.0903, Val Loss: 6.2136
Checkpoint saved!
Epoch [4/5], Train Loss: 5.4697, Val Loss: 5.8443
Checkpoint saved!
Epoch [5/5], Train Loss: 4.9349, Val Loss: 5.5590
Checkpoint saved!
Training complete. Best model saved to: causal_model_checkpoint.pth


###  Generated text based on a given prompt.


Generating text for prompt: 'What is the purpose of life?'

Generated Text:
 What is the purpose of life? of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the of the

### Attention heatmap:
The heatmap below provides insights into the model's focus during prediction.
It shows how a the built causal transformer model focuses on previous tokens when predicting outputs for each token position:
Each row represents a query token.
The attention is progressively applied to earlier tokens as you move down the heatmap.
The causal mask prevents attention to future tokens, ensuring autoregressive behavior for text generation.
This behavior is essential for models that generate text sequentially, like language models.
