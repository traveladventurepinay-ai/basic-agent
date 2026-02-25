# Session Summary - Day 6: LangChain Agents & Azure Deployment

## Part 1: Running LangChain Notebooks Locally

### basic.ipynb - LangChain ReAct Agent

**Issues Fixed:**
1. Installed packages using Python 3.10's pip (default pip pointed to Python 3.15 with no pre-built wheels)
2. Fixed `.env` loading - kernel working directory was workspace root, not notebook folder
3. Added `override=True` to prevent stale Windows system env variable from overriding real API key
4. Suppressed deprecation warnings

**Cell 1 (fixed):**
```python
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)
```

**Cell 2 (original code):**
```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool

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

agent.invoke("Add 44 and 51")
```

**Output:** `{'input': 'Add 44 and 51', 'output': '95'}`

---

### CSVDataAnalysisAgent.ipynb - CSV Analysis Agent

**Cell 1 (fixed):**
```python
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=True)
```

**Cell 2 - CSV Agent using langchain_experimental:**
```python
import pandas as pd
df = pd.DataFrame({
    "name": ["Alice","Bob","Charlie","Diana","Ethan","Fatima"],
    "department": ["AI","Data","Cloud","AI","Data","Cloud"],
    "salary": [120000, 95000, 110000, 130000, 105000, 125000],
    "years_exp": [6, 4, 8, 9, 5, 7]
})
df.to_csv("employees.csv", index=False)

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
csv_agent = create_csv_agent(
    llm,
    "employees.csv",
    verbose=True,
    allow_dangerous_code=True,
    include_df_in_prompt=False,
    number_of_head_rows=5
)

print(csv_agent.invoke({"input": "Which department has the highest average salary? Return dept and value."})["output"])
```

**Cell 3 - Custom Tools Agent:**
```python
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.openai_functions_agent.base import create_openai_functions_agent
from langchain.agents import AgentExecutor

df = pd.read_csv("employees.csv")

@tool
def list_columns() -> list[str]:
    """Return CSV column names."""
    return list(df.columns)

@tool
def head(n: int = 5) -> str:
    """Return the first n rows as a table string."""
    return df.head(n).to_string(index=False)

@tool
def describe_numeric() -> str:
    """Return pandas describe() for numeric columns."""
    return df.describe().to_string()

@tool
def groupby_mean(column: str, by: str) -> str:
    """Return mean of `column` grouped by `by` as CSV text."""
    if column not in df.columns or by not in df.columns:
        return f"Column not found. Available: {list(df.columns)}"
    out = df.groupby(by)[column].mean().reset_index().sort_values(column, ascending=False)
    return out.to_csv(index=False)

tools = [list_columns, head, describe_numeric, groupby_mean]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You analyze a local CSV using only the provided tools. "
     "Do not execute arbitrary code. If a tool is insufficient, say so."),
    MessagesPlaceholder("agent_scratchpad"),
    ("human", "{input}")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(executor.invoke({"input": "Mean salary by department (highest first) as CSV."})["output"])
```

---

## Part 2: Azure Deployment

### Package Installation Commands

```bash
# Install packages using Python 3.10 (not 3.15)
"/c/Program Files/Python310/python" -m pip install "langchain==0.3.25" "langchain-openai==0.3.12" openai
"/c/Program Files/Python310/python" -m pip install langchain_experimental

# Install Azure CLI
winget install -e --id Microsoft.AzureCLI

# Install WSL2 (run from Admin terminal)
wsl --install
```

### Files Created for Containerization

**app.py** (Flask web server wrapping the LangChain agent):
```python
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
```

**requirements.txt:**
```
langchain==0.3.25
langchain-openai==0.3.12
openai>=1.0.0,<2.0.0
python-dotenv
flask
```

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY .env .

EXPOSE 8000

CMD ["python", "app.py"]
```

**.env:**
```
OPENAI_API_KEY=sk-proj-your-key-here
TAVILY_API_KEY=tvly-dev-your-key-here
```

---

### Azure CLI Commands (Step by Step)

#### Step 1: Login to Azure
```bash
az login --tenant cc7e7d16-1fc7-4223-a441-f2828aace51f
```

#### Step 2: Build & Push Image to ACR (Cloud Build - No Docker Needed)
```bash
cd "LangChain Agents"
az acr build --registry test21 --image basic-agent:latest --no-logs .
```
- Used `--no-logs` to bypass a Windows Unicode encoding bug in Azure CLI
- Image pushed to: `test21.azurecr.io/basic-agent:latest`

#### Step 3: Enable ACR Admin Access
```bash
az acr update --name test21 --admin-enabled true
az acr credential show --name test21
```

#### Step 4: Deploy to Azure Container Instances (One-Shot Run)
```bash
az container create \
  --resource-group rg-cgajardo-6066 \
  --name basic-agent-container \
  --image test21.azurecr.io/basic-agent:latest \
  --registry-login-server test21.azurecr.io \
  --registry-username test21 \
  --registry-password "<password>" \
  --cpu 1 --memory 1.5 \
  --restart-policy Never \
  --os-type Linux

# Check logs
az container logs --resource-group rg-cgajardo-6066 --name basic-agent-container
```
- Container ran successfully, exit code 0, output: 44 + 51 = 95

#### Step 5: Deploy to Azure Container Apps (Web App)

App Service failed due to zero VM quota, so we used Container Apps (serverless):

```bash
# Register required providers
az provider register -n Microsoft.OperationalInsights --wait
az provider register -n Microsoft.App --wait

# Install Container Apps extension
az extension add --name containerapp --upgrade

# Create environment
az containerapp env create \
  --name basic-agent-env \
  --resource-group rg-cgajardo-6066 \
  --location eastus2 \
  --logs-destination none

# Deploy the app
az containerapp create \
  --name basic-agent-app \
  --resource-group rg-cgajardo-6066 \
  --environment basic-agent-env \
  --image test21.azurecr.io/basic-agent:latest \
  --registry-server test21.azurecr.io \
  --registry-username test21 \
  --registry-password "<password>" \
  --target-port 8000 \
  --ingress external \
  --cpu 0.5 --memory 1.0Gi
```

### Final Deployed App

- **URL**: https://basic-agent-app.jollysmoke-bc071a3d.eastus2.azurecontainerapps.io/
- **Ask endpoint**: https://basic-agent-app.jollysmoke-bc071a3d.eastus2.azurecontainerapps.io/ask?q=Add+44+and+51

---

## Azure Resources Created

| Resource | Type | Resource Group |
|---|---|---|
| test21 | Container Registry (ACR) | rg-cgajardo-6066 |
| basic-agent-container | Container Instance (ACI) | rg-cgajardo-6066 |
| basic-agent-env | Container Apps Environment | rg-cgajardo-6066 |
| basic-agent-app | Container App | rg-cgajardo-6066 |

## Key Troubleshooting Notes

1. **pip linked to wrong Python**: `pip` was Python 3.15 but `python` was 3.10. Used full path to Python 3.10's pip.
2. **find_dotenv() not finding .env**: Jupyter kernel cwd differs from notebook directory. Fixed with `usecwd=True`.
3. **System env variable override**: Windows had `OPENAI_API_KEY=YOUR_KEY` set. Fixed with `override=True` in `load_dotenv()`.
4. **Docker Desktop wouldn't start**: WSL2 not installed. Installed with `wsl --install` (Admin terminal). Reboot needed.
5. **az acr build Unicode crash**: Windows cp1252 encoding can't handle pip install logs. Fixed with `--no-logs` flag.
6. **App Service zero VM quota**: Subscription had no VM quota. Used Azure Container Apps (serverless) instead.
