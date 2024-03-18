# LLM PLAYGROUND

### 01-DB_FT_HF_PEFT_LORA_SAVE : 
<details>
  <summary> Fine-tuning a Sentiment Analysis Model with Parameter-Efficient Fine-Tuning (PEFT) Techniques on IMDb Data</summary>

**Imports:**<br>
Libraries for datasets, models, tokenizers, training, and evaluation from transformers, datasets, and other packages.

**Dataset:**<br>
Loading a truncated IMDb dataset from a personal repository.
Model Setup:

Initialization of the DistilBERT model (with an option for RoBERTa) for sequence classification.
Setting up label mappings for sentiment classification (Positive/Negative).

**Data Preprocessing:**<br>
Tokenization of the text data with special handling for padding tokens.

**Evaluation Setup:**<br>
Loading the accuracy metric from the evaluate library.
Defining a function to compute metrics (accuracy) for model evaluation.
Apply Untrained Model to Text:

Running the untrained model on a list of example sentences to predict their sentiment.

**Model Training:**<br>
Configuring parameters for PEFT (Parameter-efficient Fine-tuning).
Displaying trainable parameters of the model.
Setting hyperparameters for training (learning rate, batch size, epochs).
Initializing the Trainer with model, training arguments, datasets, tokenizer, and data collator.
Commencing the training process.

**Generate Prediction:**<br>
Moving the model to a specific device (MPS or CPU) for inference.
Predicting sentiments of the example sentences with the trained model.

**Push Model to Hub:**<br>
Providing two options for Hugging Face Hub login: Notebook login and key login.
Setting up identifiers for pushing the model and trainer to the Hugging Face Hub.
Optional - Load PEFT Model for Inference:

Instructions on how to load a PEFT model from the hub for inference purposes.
</details>

### 03- HF_LOAD_MODEL: 
<details>
  <summary> Text Generation Using Transformers' Pipeline with TinyLlama Model </summary>
</details>
This simple script demonstrates the use of the Transformers library for text generation by utilizing a high-level helper called 'pipeline'. It specifically employs the TinyLlama model to generate text based on a given prompt.

**Key Steps:**

- **Library Installation:**
  - Installing the `transformers` package, which provides tools for natural language processing tasks.

- **Pipeline Initialization:**
  - Importing the `pipeline` function from the Transformers library.
  - Initializing a text generation pipeline using the "TinyLlama/TinyLlama-1.1B-Chat-v1.0" model.

- **Text Generation:**
  - Using the pipeline to generate text based on the prompt "the capital of France is".<details>

### 03- : 
<details>
  <summary> X </summary>
</details><details>

### 04- : 
<details>
  <summary> X </summary>
</details><details>

### 05- : 
<details>
  <summary> X </summary>
</details><details>

### 06- : 
<details>
  <summary> X </summary>
</details><details>

### 07- : 
<details>
  <summary> X </summary>
</details><details>

### 08- : 
<details>
  <summary> X </summary>
</details><details>

### 09- : 
<details>
  <summary> X </summary>
</details><details>

### 10- : 
<details>
  <summary> X </summary>
</details><details>

### 11- : 
<details>
  <summary> X </summary>
</details><details>

### 12- : 
<details>
  <summary> X </summary>
</details><details>

### 13- : 
<details>
  <summary> X </summary>
</details><details>

### 14- : 
<details>
  <summary> X </summary>
</details><details>

### 15- : 
<details>
  <summary> X </summary>
</details><details>

### 16- : 
<details>
  <summary> X </summary>
</details><details>

### 17- : 
<details>
  <summary> X </summary>
</details><details>

### 18- : 
<details>
  <summary> X </summary>
</details><details>

### 19- : 
<details>
  <summary> X </summary>
</details><details>

### 20- : 
<details>
  <summary> X </summary>
</details><details>

