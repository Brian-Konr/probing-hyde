import transformers
import torch
from transformers import GenerationConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(filename='lm_backend.log', level=logging.ERROR,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    gen_config: dict

# Load the model and pipeline only once when the server starts
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=2
)

def make_request(message):
    return [{"role": "user", "content": message}]

@app.post("/generate")
async def generate_text(request: ChatRequest):
    try:
        outputs = pipeline(make_request(request.message), **request.gen_config)
        return outputs
    except Exception as e:
        logger.exception(f"Error in generate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
