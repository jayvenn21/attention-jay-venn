import os
import json
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="VizFold API Bridge")

# IMPORTANT: This allows your React app (running on a different port) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "test_output"

@app.get("/")
def read_root():
    return {"status": "VizFold API is live", "directory": OUTPUT_DIR}

@app.get("/meta")
def get_meta():
    """Returns the meta.json file so the frontend knows what traces exist."""
    meta_path = os.path.join(OUTPUT_DIR, "meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="meta.json not found")
    with open(meta_path, "r") as f:
        return json.load(f)

@app.get("/structure")
def get_structure():
    """Serves the raw PDB file for 3Dmol.js to render."""
    pdb_path = os.path.join(OUTPUT_DIR, "structure", "predicted.pdb")
    if not os.path.exists(pdb_path):
        raise HTTPException(status_code=404, detail="PDB structure not found")
    return FileResponse(pdb_path, media_type="text/plain")

@app.get("/tensor/{category}/{filename}")
def get_tensor(category: str, filename: str):
    """
    Dynamically loads any .pt file (attention, activations, recycling iterations),
    converts the PyTorch tensor to a standard JSON array, and returns it.
    """
    # Security check: only allow reading from specific trace directories
    if category not in ["attention", "activations", "structure_module"]:
        raise HTTPException(status_code=400, detail="Invalid trace category")
        
    filepath = os.path.join(OUTPUT_DIR, "trace", category, f"{filename}.pt")
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Tensor {filename}.pt not found")
        
    try:
        # Load the PyTorch file safely
        tensor = torch.load(filepath, weights_only=True)
        
        # Convert to float32 (if fp16), detach, move to CPU, and convert to nested Python list
        tensor_list = tensor.float().detach().cpu().numpy().tolist()
        
        return {
            "name": filename,
            "shape": list(tensor.shape),
            "data": tensor_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing tensor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Runs the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)