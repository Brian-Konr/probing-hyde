# import transformers
# from transformers import GenerationConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

from prediction_function_layers import get_embedding_from_generation, probe_generation

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
# probe_name = "meta-llama_Llama-3.1-8B-Instruct_4_all-conj.pt"
probe_name = "meta-llama_Llama-3.1-8B-Instruct_all-conj_ATTSE1DCNN_head8_dropout0.1.pt"
probe_path = os.path.join(os.getcwd(), probe_name)
# layer = -4
 

####################################################################################
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device=2
# )

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto"
# )

# def make_request(message):
#     return [{"role": "user", "content": message}]

# @app.post("/generate")
# async def generate_text(request: ChatRequest):
#     try:
#         outputs = pipeline(make_request(request.message), **request.gen_config)
#         return outputs
#     except Exception as e:
#         logger.exception(f"Error in generate_text: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
####################################################################################

try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True
    )
    model.eval()
except Exception as e:
    logger.exception(f"Error loading model or tokenizer: {str(e)}")
    raise e

@app.post("/generate")
async def generate_text(request: ChatRequest):
    try:
        results = get_embedding_from_generation(
            message=request.message,
            gen_config=request.gen_config,
            model=model,
            tokenizer=tokenizer,
            device=model.device,
            # layer=layer  
        )
        
        response = probe_generation(
            probe_path=probe_path,
            probe_name=probe_name,
            model=model,
            tokenizer=tokenizer,
            results=results
        )
        return response
    except Exception as e:
        logger.exception(f"Error in generate_text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
