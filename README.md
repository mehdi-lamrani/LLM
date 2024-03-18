# LLM PLAYGROUND

# 01-DB_FT_HF_PEFT_LORA_SAVE : 
<details>
  <summary>### Fine-tuning a Sentiment Analysis Model with Parameter-Efficient Fine-Tuning (PEFT) Techniques on IMDb Dat</summary>

**Imports:**
Libraries for datasets, models, tokenizers, training, and evaluation from transformers, datasets, and other packages.

**Dataset:**
Loading a truncated IMDb dataset from a personal repository.
Model Setup:

Initialization of the DistilBERT model (with an option for RoBERTa) for sequence classification.
Setting up label mappings for sentiment classification (Positive/Negative).

**Data Preprocessing:**
Tokenization of the text data with special handling for padding tokens.

**Evaluation Setup:**
Loading the accuracy metric from the evaluate library.
Defining a function to compute metrics (accuracy) for model evaluation.
Apply Untrained Model to Text:

Running the untrained model on a list of example sentences to predict their sentiment.

**Model Training:**
Configuring parameters for PEFT (Parameter-efficient Fine-tuning).
Displaying trainable parameters of the model.
Setting hyperparameters for training (learning rate, batch size, epochs).
Initializing the Trainer with model, training arguments, datasets, tokenizer, and data collator.
Commencing the training process.

**Generate Prediction:**
Moving the model to a specific device (MPS or CPU) for inference.
Predicting sentiments of the example sentences with the trained model.

**Push Model to Hub:**
Providing two options for Hugging Face Hub login: Notebook login and key login.
Setting up identifiers for pushing the model and trainer to the Hugging Face Hub.
Optional - Load PEFT Model for Inference:

Instructions on how to load a PEFT model from the hub for inference purposes.
</details>
