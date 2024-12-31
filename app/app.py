from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import logging
import time
import torch.nn.functional as F
import os
import psutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def load_model():
    global tokenizer, model, id2label
    try:
        model_name_or_path = "guilhermeengineai/ner-test"

        huggingface_token = os.getenv("HF_TOKEN")
        if not huggingface_token:
            raise ValueError("HF_TOKEN variable not set.")
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 ** 2)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_auth_token=huggingface_token
        )

        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            use_auth_token=huggingface_token,
            device_map="auto"
        )

        model.eval()

        id2label = model.config.id2label
        mem_after = process.memory_info().rss / (1024 ** 2)  # in MB
        memory_diff = mem_after - mem_before
        logging.info(f"Model Memory Footprint: {memory_diff:.2f} MB")

        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model during startup.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Here goes everything that happens on loading phase
    load_model()
    yield
    # Here goes everything that happens on shutdown phase

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"message": "Hello world!"}

def split_into_words(text):
    return text.split()

class Input(BaseModel):
    text: str

@app.post("/predict")
def predict(request: Input):
    try:
        start_time = time.perf_counter()

        words = split_into_words(request.text)
        if not words:
            raise ValueError("Input text is empty or could not be split into words.")

        tokenized_inputs = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            outputs = model(**tokenized_inputs)

        inference_time = (time.perf_counter() - start_time) * 1000

        probabilities = F.softmax(outputs.logits, dim=-1)

        # Get the highest probability and corresponding label for each token
        max_probs, pred_labels = torch.max(probabilities, dim=-1)

        tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
        word_ids = tokenized_inputs.word_ids(batch_index=0)

        entities = []
        current_entity = None
        entity_scores = []
        last_word_idx = -1

        for idx, (token_id, prob) in enumerate(zip(pred_labels[0], max_probs[0])):
            word_idx = word_ids[idx]

            if word_idx is None:
                continue  # Skip special tokens

            if word_idx == last_word_idx:
                continue  # Skip tokens that belong to the same word as the previous token

            last_word_idx = word_idx

            label = id2label[token_id.item()]
            word = words[word_idx]
            confidence = prob.item()

            if label.startswith("B-"):
                if current_entity:
                    # Finalize the previous entity
                    avg_confidence = sum(entity_scores) / len(entity_scores)
                    current_entity["confidence"] = round(avg_confidence, 4)
                    entities.append(current_entity)
                # Start a new entity
                current_entity = {
                    "word": word,
                    "type": label[2:],
                    "start": word_idx,
                    "end": word_idx
                }
                entity_scores = [confidence]
            elif label.startswith("I-") and current_entity and current_entity["type"] == label[2:]:
                # Continue the current entity
                current_entity["word"] += " " + word
                current_entity["end"] = word_idx
                entity_scores.append(confidence)
            else:
                # End of the current entity
                if current_entity:
                    avg_confidence = sum(entity_scores) / len(entity_scores)
                    current_entity["confidence"] = round(avg_confidence, 4)
                    entities.append(current_entity)
                    current_entity = None
                    entity_scores = []

        # Append the last entity if it's still open
        if current_entity:
            avg_confidence = sum(entity_scores) / len(entity_scores)
            current_entity["confidence"] = round(avg_confidence, 4)
            entities.append(current_entity)

        logging.info(
            f"Input: {request.text}, Dets: {entities}, Inf. Time: {inference_time:.2f}ms."
        )

        response_entities = [
            {
                "type": entity["type"],
                "start": entity["start"],
                "end": entity["end"]
            }
            for entity in entities
        ]

        return {
            "text": request.text,
            "entities": response_entities
        }

    except Exception as e:
        logging.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    