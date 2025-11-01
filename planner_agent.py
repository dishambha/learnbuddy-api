# planner_agent.py
from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq

# Create the FastAPI app
app = FastAPI(title="Planner Agent API", version="1.0")

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY") 

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")  # ← Replace this with your real API key

# Define the input model
class PlannerRequest(BaseModel):
    query: str

@app.post("/plan/")
async def generate_plan(request: PlannerRequest):
    """Generate a detailed plan using Groq's LLM."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast Groq model
            messages=[
                {"role": "system", "content": "You are a helpful planning assistant."},
                {"role": "user", "content": request.query}
            ]
        )
        response = completion.choices[0].message.content
        return {"plan": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Planner Agent running with Groq API 🚀"}