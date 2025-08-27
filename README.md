# ReAct Agent for Gemini

## Project Description
This project implements a ReAct (Reasoning and Acting) agent designed to interact with Google's Gemini Large Language Model (LLM). The agent is capable of processing user queries, reasoning about the best course of action, and utilizing external tools (like Google Search and Wikipedia) to gather information before formulating a comprehensive answer. It exposes a Flask API for seamless interaction.

## Features
*   **ReAct Pattern Implementation**: Leverages the ReAct (Reasoning and Acting) prompting strategy for enhanced LLM capabilities.
*   **Gemini LLM Integration**: Optimized for interaction with Google's Gemini models via Vertex AI.
*   **Tool Use**: Integrates external tools such as Google Search (via SerpAPI) and Wikipedia for information retrieval.
*   **Conversation Memory**: Maintains a history of interactions to provide context-aware responses.
*   **Caching**: Implements an in-memory cache for tool results to reduce redundant API calls and improve performance.
*   **Repetition Detection**: Prevents infinite loops by detecting repetitive agent actions.
*   **Structured Logging and Tracing**: Provides detailed logs and traces of the agent's thought process and actions.
*   **Flask API**: Exposes a RESTful API for easy integration with frontend applications.
*   **Pydantic Models**: Uses Pydantic for robust data validation and serialization of agent states, messages, and tool responses.

## Architecture
The ReAct agent operates on a loop of "Thought," "Action," and "Observation."

```mermaid
graph TD
    A[User Query] --> B{Agent.execute()};
    B --> C{Agent.think()};
    C --> D[Construct Prompt with Query, History, Tools];
    D --> E[Gemini LLM];
    E --> F[LLM Response (Thought/Action/Answer)];
    F --> G{Agent.parse_response()};
    G --> H{Agent.decide()};
    H -- If Action Suggested --> I{Agent.act()};
    I --> J[Execute Tool (e.g., Google Search, Wikipedia)];
    J --> K[Tool Observation];
    K --> L[Add Observation to Memory];
    L --> C;
    H -- If Final Answer --> M[Agent.final_answer];
    M --> N[Return Response to User];
    H -- If Continue Thinking --> C;
```

### Components:
*   **Agent**: The core orchestrator, managing the ReAct loop, state, memory, and tool interactions.
*   **Gemini LLM**: The brain of the agent, responsible for generating thoughts, deciding on actions, and formulating answers.
*   **Tools**: External services (e.g., SerpAPI for Google Search, Wikipedia API) that the agent can invoke to gather information.
*   **MemoryManager**: Stores and manages the conversation history.
*   **ToolCache**: Caches results from tool invocations to optimize performance.
*   **RepetitionDetector**: Monitors agent actions to prevent repetitive loops.
*   **Flask API**: Provides the interface for external applications to interact with the agent.

## Project Structure

```
react-agent/
├── config/
│   ├── config.yml             # Main project configuration
│   └── credentials/
│       ├── key.json           # Gemini API credentials (example)
│       └── key.yml            # SerpAPI key (example)
├── data/
│   ├── input/
│   │   └── react.txt          # Prompt template for the ReAct agent
│   └── output/
│       └── trace.txt          # Agent execution trace logs
├── logs/
│   └── app.log                # Application logs
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   ├── logging.py         # Logging configuration
│   │   └── setup.py           # Project setup (e.g., Vertex AI initialization)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemini.py          # Gemini LLM interaction logic
│   ├── react/
│   │   ├── __init__.py
│   │   └── agent.py           # Core ReAct agent implementation and Flask API
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── manager.py         # (Potentially deprecated/alternative) Tool manager
│   │   ├── serp.py            # Google Search tool (SerpAPI integration)
│   │   └── wiki.py            # Wikipedia search tool
│   └── utils/
│       ├── __init__.py
│       └── io.py              # Utility functions for file I/O
├── .gitignore                 # Git ignore file
├── poetry.lock                # Poetry lock file
├── pyproject.toml             # Project and dependency definitions (Poetry)
└── README.md                  # This README file
```

## Setup and Installation

### Prerequisites
*   Python 3.10+
*   Poetry (for dependency management)
*   Google Cloud Project with Vertex AI API enabled
*   SerpAPI Key (for Google Search functionality)

### 1. Clone the repository
```bash
git clone https://github.com/your-repo/react-agent.git
cd react-agent
```

### 2. Install dependencies using Poetry
```bash
poetry install
poetry shell
```

### 3. Configure Google Cloud Credentials
Ensure your Google Cloud credentials are set up. You can either:
*   Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to the path of your service account key JSON file.
*   Place your service account key JSON file at `./credentials/key.json` (as configured in `config/config.yml`).

### 4. Configure SerpAPI Key
Obtain a SerpAPI key from [serpapi.com](https://serpapi.com/).
Create or update the `./credentials/key.yml` file with your SerpAPI key:
```yaml
serp:
  key: YOUR_SERPAPI_KEY
```

### 5. Initialize Vertex AI
The `src/config/setup.py` file handles Vertex AI initialization. Ensure your `project_id` and `region` are correctly set in `config/config.yml`.

## Configuration
The main configuration is located in `config/config.yml`:

```yaml
project_id: your-gcp-project-id
credentials_json: ./credentials/key.json
region: us-central1
model_name: gemini-2.5-flash # Or other Gemini models like gemini-pro
```

You can also adjust agent-specific parameters within `src/react/agent.py` in the `create_agent` function, such as `max_iterations`, `confidence_threshold`, `cache_ttl`, and `max_history_size`.

## Usage

### Running the Flask API
To start the ReAct agent API server:
```bash
python src/react/agent.py
```
The server will run on `http://0.0.0.0:5000` (default).

### Interacting with the API
You can interact with the agent using `curl` or any API client.

#### Ask a Query
**Endpoint**: `POST /ask`
**Body**: `{"query": "Your question here"}`

Example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query": "Who is the current president of the United States?"}' http://localhost:5000/ask
```

#### Get Conversation History
**Endpoint**: `GET /api/conversations`

Example:
```bash
curl http://localhost:5000/api/conversations
```

## API Endpoints

*   `POST /ask`: Submits a query to the ReAct agent and receives a response.
    *   **Request Body**: `{"query": "string"}`
    *   **Response Body**: `{"response": "string", "status": "success" | "error"}`
*   `GET /api/conversations`: Retrieves the current conversation history.
    *   **Response Body**: `[{"role": "string", "content": "string", "timestamp": "ISO 8601 string"}]`

## Tools

The agent is equipped with the following tools:

*   **Google Search (SerpAPI)**:
    *   **Description**: Performs web searches for current information and general web results.
    *   **Usage**: Used for queries requiring up-to-date information, specific facts, or broad web content.
    *   **Implementation**: `src/tools/serp.py`
*   **Wikipedia Search**:
    *   **Description**: Fetches factual information, biographies, and summaries from Wikipedia.
    *   **Usage**: Ideal for historical facts, definitions, and detailed information on known entities.
    *   **Implementation**: `src/tools/wiki.py`

## Logging and Tracing
The agent provides comprehensive logging and tracing to help understand its decision-making process.
*   **Application Logs**: `logs/app.log`
*   **Execution Traces**: `data/output/trace.txt` (records the agent's thoughts, actions, and observations for each iteration).

## Contributing
Contributions are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.