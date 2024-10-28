import transformers
import torch
from transformers import GenerationConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
    device_map="cuda:0",
)

def make_request(message):
    return [{"role": "user", "content": message}]

@app.post("/generate")
async def generate_text(request: ChatRequest):
    try:
        pipeline.generation_config = GenerationConfig(**request.gen_config)
        outputs = pipeline(make_request(request.message))
        return outputs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 