from fastapi import FastAPI, Request
from pydantic import BaseModel
from groq import Groq
import os
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, HTTPException
# ... (rest of your imports)

# --- Initialize app ONCE ---
app = FastAPI(title="LearnBuddy Multi-Agent API")

# --- Use an environment variable for your key! ---
# It's safer than hardcoding it.
# Set this in your terminal: export GROQ_API_KEY="your_real_key"
# THIS IS THE FIX
api_key = os.getenv("GROQ_API_KEY") 

if not api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")
groq_client = Groq(api_key=api_key)

# --- Delete this line, it's a duplicate ---
# app = FastAPI() 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class PlannerRequest(BaseModel):
    query: str

class SubjectRequest(BaseModel):
    subject: str

class FeedbackRequest(BaseModel):
    input: str

class MotivationRequest(BaseModel):
    mood: str


# ---------- Agents ----------
@app.post("/planner")
async def planner_agent(payload: PlannerRequest):
    try:
        # --- 1. VALIDATION STEP ---
        # We ask a fast model to classify the input query first.
        validation_prompt = f"""
        Analyze the user's query: "{payload.query}"
        
        Is this query a valid educational subject, academic topic, or a skill that someone can create a learning plan for?
        
        Respond in JSON format with two keys:
        1. "is_topic": boolean (true if it is a valid topic, false otherwise)
        2. "reason": string (a brief explanation for your decision)
        
        Examples:
        - Query: "Quantum Physics" -> {{"is_topic": true, "reason": "This is a valid field of study."}}
        - Query: "How to bake bread" -> {{"is_topic": true, "reason": "This is a valid skill to learn."}}
        - Query: "asdfghjkl" -> {{"is_topic": false, "reason": "This appears to be random nonsense."}}
        - Query: "what is the time" -> {{"is_topic": false, "reason": "This is a question, not a learning topic."}}
        """
        
        validation_response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Use the fast model for validation
            messages=[{"role": "user", "content": validation_prompt}],
            max_tokens=150,
            response_format={"type": "json_object"}  # Force the model to output JSON
        )
        
        # Parse the JSON response from the validation model
        validation_result = json.loads(validation_response.choices[0].message.content)

        # --- 2. CHECK THE RESULT ---
        if not validation_result.get("is_topic"):
            # If it's NOT a topic, return a 400 Bad Request error
            raise HTTPException(
                status_code=400, 
                detail=f"Input is not a valid learning topic. Reason: {validation_result.get('reason', 'Invalid input')}"
            )

        # --- 3. PLANNER STEP (if validation passed) ---
        # If is_topic was true, we proceed with the big model
        
        planner_prompt = f"""
        Act as an expert learning advisor. A user wants to learn the following subject: '{payload.query}'.

        Create a comprehensive, step-by-step learning roadmap for them to master this topic.
        
        Your plan must:
        1.  Break down the subject into logical modules or sections (e.g., "Module 1: The Basics", "Module 2: Core Concepts", etc.).
        2.  For each module, list the key concepts or skills they need to learn.
        3.  Suggest a realistic study schedule, such as how many hours to dedicate per week and how long each module might take.
        4.  Provide a clear path that guides them from the fundamentals to more advanced topics.
        
        Make the roadmap encouraging and easy for a beginner to follow.
        """
        
        planner_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Use the powerful model for the plan
            messages=[{"role": "user", "content": planner_prompt}],
            max_tokens=1024,
        )
        
        return {"plan": planner_response.choices[0].message.content}

    except HTTPException as he:
        # This makes sure our 400 error is sent correctly
        raise he
    except Exception as e:
        print("Error in planner_agent:", e)
        # This catches any other errors (e.g., Groq API is down)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/subject")
# CHANGE THIS: Use the Pydantic model 'SubjectRequest'
async def subject_agent(payload: SubjectRequest):
    try:
        prompt = f"Explain the subject {payload.subject} in a concise, beginner-friendly way with examples."
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return {"subject_explanation": response.choices[0].message.content}
    except Exception as e:
        print("Error in subject_agent:", e)
        return {"error": str(e)}


@app.post("/feedback")
# CHANGE THIS: Use the Pydantic model 'FeedbackRequest'
async def feedback_agent(payload: FeedbackRequest):
    try:
        prompt = f"Provide constructive feedback for this student response: {payload.input}"
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return {"feedback": response.choices[0].message.content}
    except Exception as e:
        print("Error in feedback_agent:", e)
        return {"error": str(e)}


@app.post("/motivation")
# CHANGE THIS: Use the Pydantic model 'MotivationRequest'
async def motivation_agent(payload: MotivationRequest):
    try:
        prompt = f"Motivate a learner who is feeling {payload.mood}. Give a short, encouraging message."
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        return {"motivation": response.choices[0].message.content}
    except Exception as e:
        print("Error in motivation_agent:", e)
        return {"error": str(e)}


@app.get("/")
async def home():
    return {"message": "Welcome to LearnBuddy Multi-Agent API!"}