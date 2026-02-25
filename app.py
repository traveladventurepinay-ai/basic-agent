import warnings
warnings.filterwarnings("ignore")

import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool

load_dotenv()

app = Flask(__name__)

llm = ChatOpenAI(model="gpt-4o-mini")

def add(a, b):
    return int(a) + int(b)

tools = [
    Tool(
        name="add_numbers",
        func=lambda q: add(q.split()[0], q.split()[1]),
        description="Add two numbers. Input format: a b"
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

@app.route("/")
def home():
    return "LangChain Agent is running! Use /ask?q=your+question"

@app.route("/ask")
def ask():
    query = request.args.get("q", "Add 44 and 51")
    result = agent.invoke(query)
    return jsonify({"input": result["input"], "output": result["output"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
