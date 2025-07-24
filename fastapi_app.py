# pip install fastapi uvicorn langchain-ollama python-multipart
import os, sqlite3, json, re
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
import plotly.express as px
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# ------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------
DB_PATH   = "data.db"
OLLAMA_URL = "http://127.0.0.1:11434"
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

gemini_llm = GoogleGenerativeAI(api_key=GEMINI_KEY, model="gemini-2.5-flash")
ollama_llm = Ollama(base_url=OLLAMA_URL, model="qwen2.5:7b")

# ------------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------------
app = FastAPI(title="AI Data Query Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str
    visualize: bool = False
    provider: Literal["gemini", "ollama"] = "gemini"

# ------------------------------------------------------------------
# Helpers (shared with ai_agent.py)
# ------------------------------------------------------------------
def extract_sql(raw: str) -> str:
    """Extract SQL query from LLM response with multiple fallback patterns"""
    if not raw or not raw.strip():
        raise ValueError("Empty SQL response from LLM")
    
    # Pattern 1: SQL code blocks
    patterns = [
        r"```sql\n(.*?)\n```",
        r"```\n(SELECT.*?)\n```",
        r"```(SELECT.*?)```"
    ]
    
    for pattern in patterns:
        m = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        if m:
            sql = m.group(1).strip()
            if sql and not sql.lower() == 'select':
                return sql
    
    # Pattern 2: Lines starting with specific prefixes
    prefixes = ["sqlquery:", "sql:", "query:"]
    for line in raw.splitlines():
        line_lower = line.strip().lower()
        for prefix in prefixes:
            if line_lower.startswith(prefix):
                sql = line.split(":", 1)[-1].strip()
                if sql and not sql.lower() == 'select':
                    return sql
    
    # Pattern 3: Direct SQL statement
    if raw.strip().lower().startswith("select") and len(raw.strip()) > 6:
        return raw.strip()
    
    # Pattern 4: Find any SELECT statement in the text
    select_match = re.search(r'(SELECT\s+.*?(?:;|$))', raw, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip().rstrip(';')
        if sql and not sql.lower() == 'select':
            return sql
    
    # If all else fails, return the raw response but log it
    print(f"Warning: Could not extract proper SQL from: {raw}")
    raise ValueError(f"Could not extract valid SQL from LLM response: {raw[:200]}...")

def answer(question: str, provider: str, visualize: bool):
    try:
        print(f"Using provider: {provider}")
        llm = gemini_llm if provider == "gemini" else ollama_llm
        print(f"LLM selected: {type(llm)}")
        
        chain = create_sql_query_chain(llm, db)
        print("SQL chain created")

        raw_sql = chain.invoke({"question": question})
        print(f"Raw SQL response: {raw_sql}")
        
        sql = extract_sql(raw_sql)
        print(f"Extracted SQL: {sql}")

        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(sql, conn)
        print(f"Query executed, rows returned: {len(df)}")

        chart_base64 = None
        if visualize and not df.empty:
            try:
                import base64
                import io
                
                # Smart chart selection based on data
                if len(df.columns) >= 2:
                    # Check if we have numeric data for better visualization
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    
                    if len(numeric_cols) >= 1:
                        # If we have numeric data, create appropriate chart
                        x_col = df.columns[0]
                        y_col = numeric_cols[0] if numeric_cols[0] != x_col else (numeric_cols[1] if len(numeric_cols) > 1 else df.columns[1])
                        
                        # Choose chart type based on data
                        if df[x_col].dtype == 'object' or len(df[x_col].unique()) < 20:
                            # Categorical data - use bar chart
                            fig = px.bar(df, x=x_col, y=y_col, 
                                       title=f"Chart for: {question}",
                                       color_discrete_sequence=['#4f46e5'])
                        else:
                            # Continuous data - use line chart
                            fig = px.line(df, x=x_col, y=y_col,
                                        title=f"Chart for: {question}",
                                        color_discrete_sequence=['#4f46e5'])
                    else:
                        # No numeric data, create a simple count chart
                        fig = px.bar(df, x=df.columns[0], y=df.columns[1],
                                   title=f"Chart for: {question}",
                                   color_discrete_sequence=['#4f46e5'])
                    
                    # Improve chart styling
                    fig.update_layout(
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        showlegend=False
                    )
                    
                    # Convert to base64 instead of saving to file
                    img_bytes = fig.to_image(format="png", width=800, height=500, scale=2)
                    chart_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    print(f"Chart created as base64 (length: {len(chart_base64)})")
                else:
                    print("Not enough columns for visualization")
            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                print(f"Traceback: {traceback.format_exc()}")
                # Continue without chart

        return {
            "sql": sql,
            "table": df.to_dict(orient="records"),
            "chart_base64": chart_base64
        }
    except Exception as e:
        print(f"Error in answer function: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise

# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/health")
def health_check():
    try:
        # Test database connection
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            # Get sample data from first table if exists
            sample_data = {}
            if tables:
                table_name = tables[0][0]
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
                rows = cursor.fetchall()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = [col[1] for col in cursor.fetchall()]
                sample_data = {
                    "table": table_name,
                    "columns": columns,
                    "sample_rows": rows
                }
        
        # Test API key
        api_key_status = "present" if GEMINI_KEY else "missing"
        
        return {
            "status": "healthy",
            "database": f"connected, tables: {[t[0] for t in tables]}",
            "sample_data": sample_data,
            "gemini_api_key": api_key_status,
            "ollama_url": OLLAMA_URL
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/test-sql")
def test_sql():
    """Test endpoint to manually check SQL generation"""
    try:
        test_question = "Show me all data"
        llm = gemini_llm
        chain = create_sql_query_chain(llm, db)
        raw_sql = chain.invoke({"question": test_question})
        
        return {
            "question": test_question,
            "raw_response": raw_sql,
            "extracted_sql": extract_sql(raw_sql) if raw_sql else "No response"
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}
@app.post("/query")
def query_endpoint(req: QueryRequest):
    try:
        print(f"Received query: {req.question}, provider: {req.provider}, visualize: {req.visualize}")
        result = answer(req.question, req.provider, req.visualize)
        print(f"Query successful: {result}")
        return result
    except Exception as e:
        print(f"Error in query_endpoint: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve the generated chart
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="."), name="static")