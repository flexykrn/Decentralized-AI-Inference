from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import time

app = FastAPI(title="DiCAI OpenAI Proxy")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "default")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # TODO: Forward to inference backend
    response = {
        "id": f"chatcmpl-{hash(str(messages))}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": f"[DiCAI] {model} response placeholder",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    
    if stream:
        async def generate():
            yield f"data: {json.dumps({'choices': [{'delta': {'content': response['choices'][0]['message']['content']}}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    return response

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "llama-3-8b", "object": "model", "owned_by": "dicai"},
            {"id": "llama-3-70b", "object": "model", "owned_by": "dicai"},
        ],
    }

@app.get("/health")
async def health():
    return {"status": "ok", "service": "dicai-proxy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
