| Model    | Type                                  | Strengths                                                    | Weaknesses                                                                        | Typical Use Case                                             |
| -------- | ------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **LSTM** | Recurrent Neural Network              | Good for sequential data, lightweight                        | Limited context (long-range dependencies), slower on long sequences               | Small datasets, simple NLP tasks                             |
| **BERT** | Transformer (Encoder)                 | Captures full context (bidirectional), pretrained embeddings | Heavier, requires fine-tuning                                                     | Classification, NER, QA, sentiment analysis                  |
| **GPT**  | Transformer (Decoder, Autoregressive) | Excellent language understanding and generation              | Mainly designed for generation, needs prompting or fine-tuning for classification | Text generation, zero-shot/few-shot classification, chatbots |


| Step               | LSTM                                     | BERT                                      | GPT                                                          |
| ------------------ | ---------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| Tokenization       | `Tokenizer` + sequences                  | `BertTokenizer` + attention masks         | `GPT2Tokenizer` + attention masks, pad with `eos_token`      |
| Model Architecture | Embedding → LSTM → Dense                 | Pretrained BERT encoder → Pooling → Dense | Pretrained GPT decoder → Classification head                 |
| Training           | From scratch (embedding + LSTM)          | Fine-tune pretrained model                | Fine-tune pretrained model                                   |
| Prediction         | Predict after padding → sequence → model | Directly on tokenized text                | Zero-shot with prompts **or** fine-tuned classification head |


LSTM

# Tokenize & pad sequences
tokenizer = Tokenizer()
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=max_len)
# Model
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_len),
    LSTM(32),
    Dense(1, activation="sigmoid")
])




BERT 

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


GPT

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
