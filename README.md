# University Chatbot
AI-powered chatbot designed to handle university-related queries using a simple neural network built with TensorFlow and deployed using Flask.

# Project Workflow
### 1. Load Dataset
Use manually created dataset from dialogue.json file

Each entry in the dialogue list represents a conversation intent or topic, with the following fields:
- label: A unique identifier for the intent (e.g., "greeting", "tuition_fees").
- user: A list of sample user inputs that map to this intent.
- responses: A list of one or more possible responses the chatbot should give.

### 2. Data Preprocessing
1. Tokenization
2. Sequence creation
3. Padding
4. Label encoding

### 3. Model Architecture
- Embedding Layer – Converts words to dense vectors
- GlobalAveragePooling1D – Aggregates information across the sequence
- Dense Layers – Extracts intent-specific features
- Output Layer – Softmax classifier for predicting the intent

### 4. Model Inference
Create function that takes in user query, preprocesses it, sends it to model and returns model's answer.

### 5. Saved Files
Saves tokenizer, label encoder, and model

### 6. Model Deployment (app.py)
- Load tokenizer, label encoder and the model.
- Deploy our model using Flask
- Use CORS for frontend integration
- Modify model inference code to recieve user question from front end and return model answer.
