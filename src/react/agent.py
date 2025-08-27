from flask import Flask, request, jsonify
from flask_cors import CORS
from vertexai.generative_models import GenerativeModel, Part
from tools.serp import search as google_search
from tools.wiki import search as wiki_search
from utils.io import write_to_file
from config.logging import logger
from config.setup import config
from llm.gemini import generate
from utils.io import read_file
from pydantic import BaseModel, Field, validator
from typing import Callable, Union, List, Dict, Optional, Any
from enum import Enum, auto
from dataclasses import dataclass
import json
import time
import hashlib
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Configuration constants
PROMPT_TEMPLATE_PATH = "./data/input/react.txt"
OUTPUT_TRACE_PATH = "./data/output/trace.txt"
DEFAULT_MAX_ITERATIONS = 10
DEFAULT_CACHE_TTL = 300  # 5 minutes
DEFAULT_MAX_HISTORY_SIZE = 20

class Name(Enum):
    """Enumeration for tool names available to the agent."""
    WIKIPEDIA = auto()
    GOOGLE = auto()
    NONE = auto()

    def __str__(self) -> str:
        return self.name.lower()

class AgentState(Enum):
    """Enumeration for agent execution states."""
    INITIALIZING = auto()
    THINKING = auto()
    ACTING = auto()
    DECIDING = auto()
    COMPLETED = auto()
    ERROR = auto()
    MAX_ITERATIONS_REACHED = auto()

@dataclass
class ExecutionMetrics:
    """Tracks execution metrics for the agent."""
    start_time: float
    iterations: int = 0
    tool_calls: int = 0
    errors: int = 0
    cache_hits: int = 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

class ToolMetadata(BaseModel):
    """Metadata for tool registration."""
    name: Name
    description: str
    usage_examples: List[str] = Field(default_factory=list)
    max_retries: int = 3
    timeout: int = 30
    
    class Config:
        use_enum_values = True

class Choice(BaseModel):
    """Represents a choice of tool with validation."""
    name: Name = Field(..., description="The name of the tool chosen.")
    reason: str = Field(..., description="The reason for choosing this tool.")
    confidence: float = Field(default=0.0, description="Confidence score for the choice.")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class Message(BaseModel):
    """Represents a message with enhanced metadata."""
    role: str = Field(..., description="The role of the message sender.")
    content: str = Field(..., description="The content of the message.")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ActionResponse(BaseModel):
    """Structured response for agent actions."""
    action: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    thought: Optional[str] = None
    confidence: float = 0.0
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class ToolCache:
    """Simple in-memory cache for tool results."""
    
    def __init__(self, ttl: int = DEFAULT_CACHE_TTL):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def _get_key(self, tool_name: str, query: str) -> str:
        """Generate cache key for tool and query."""
        return hashlib.md5(f"{tool_name}:{query}".encode()).hexdigest()
    
    def get(self, tool_name: str, query: str) -> Optional[Any]:
        """Get cached result if available and not expired."""
        key = self._get_key(tool_name, query)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry['timestamp'] < self.ttl:
                return entry['result']
            else:
                del self.cache[key]
        return None
    
    def set(self, tool_name: str, query: str, result: Any) -> None:
        """Cache tool result."""
        key = self._get_key(tool_name, query)
        self.cache[key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

class Tool:
    """Enhanced tool wrapper with metadata and caching."""

    def __init__(self, metadata: ToolMetadata, func: Callable[[str], str], cache: ToolCache):
        self.metadata = metadata
        self.func = func
        self.cache = cache
        self.call_count = 0
        self.error_count = 0
        self.last_used = None

    def use(self, query: str) -> str:
        """Execute tool with caching and error handling."""
        self.call_count += 1
        self.last_used = datetime.now()
        
        # Check cache first
        cached_result = self.cache.get(str(self.metadata.name), query)
        if cached_result is not None:
            logger.info(f"Cache hit for {self.metadata.name} with query: {query}")
            return cached_result
        
        # Execute with retries
        for attempt in range(self.metadata.max_retries):
            try:
                result = self.func(query)
                # Cache successful result
                self.cache.set(str(self.metadata.name), query, result)
                return result
            except Exception as e:
                self.error_count += 1
                logger.error(f"Error executing tool {self.metadata.name} (attempt {attempt + 1}): {e}")
                if attempt == self.metadata.max_retries - 1:
                    return str(e)
                time.sleep(1)  # Brief delay before retry
        
        return "Tool execution failed after all retries"

class MemoryManager:
    """Manages conversation memory with summarization."""
    
    def __init__(self, max_size: int = DEFAULT_MAX_HISTORY_SIZE):
        self.max_size = max_size
        self.messages: List[Message] = []
        self.summary = ""
    
    def add_message(self, message: Message) -> None:
        """Add message and manage memory size."""
        self.messages.append(message)
        
        if len(self.messages) > self.max_size:
            # Simple truncation strategy - in production, use summarization
            self.messages = self.messages[-self.max_size:]
            logger.info(f"Truncated message history to {self.max_size} messages")
    
    def get_recent_history(self, n: int = None) -> List[Message]:
        """Get recent messages."""
        if n is None:
            return self.messages
        return self.messages[-n:]
    
    def get_formatted_history(self) -> str:
        """Get formatted history string."""
        return "\n".join([f"{msg.role}: {msg.content}" for msg in self.messages])

class RepetitionDetector:
    """Detects repetitive patterns in agent behavior."""
    
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.recent_actions: List[str] = []
    
    def add_action(self, action: str) -> None:
        """Add action to history."""
        self.recent_actions.append(action)
        if len(self.recent_actions) > self.window_size * 2:
            self.recent_actions = self.recent_actions[-self.window_size * 2:]
    
    def is_repetitive(self) -> bool:
        """Check if recent actions are repetitive."""
        if len(self.recent_actions) < self.window_size:
            return False
        
        recent = self.recent_actions[-self.window_size:]
        return len(set(recent)) == 1  # All actions are the same

class Agent:
    """Enhanced agent with better state management and error handling."""

    def __init__(self, model: GenerativeModel, config: Dict[str, Any] = None):
        self.model = model
        self.config = config or {}
        self.tools: Dict[Name, Tool] = {}
        self.memory = MemoryManager(self.config.get('max_history_size', DEFAULT_MAX_HISTORY_SIZE))
        self.cache = ToolCache(self.config.get('cache_ttl', DEFAULT_CACHE_TTL))
        self.repetition_detector = RepetitionDetector()
        self.state = AgentState.INITIALIZING
        self.metrics = ExecutionMetrics(start_time=time.time())
        
        # Configuration
        self.max_iterations = self.config.get('max_iterations', DEFAULT_MAX_ITERATIONS)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.template = self.load_template()
        
        # State tracking
        self.query = ""
        self.current_iteration = 0
        self.final_answer = None

    def load_template(self) -> str:
        """Load prompt template with error handling."""
        try:
            return read_file(PROMPT_TEMPLATE_PATH)
        except Exception as e:
            logger.error(f"Failed to load template: {e}")
            return "Query: {query}\nHistory: {history}\nTools: {tools}\nPlease provide a response."

    def register_tool(self, metadata: ToolMetadata, func: Callable[[str], str]) -> None:
        """Register a tool with metadata."""
        self.tools[metadata.name] = Tool(metadata, func, self.cache)
        logger.info(f"Registered tool: {metadata.name} - {metadata.description}")

    def trace(self, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Enhanced tracing with metadata."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        
        if role != "system":
            self.memory.add_message(message)
        
        # Write to trace file
        try:
            trace_content = f"[{message.timestamp.isoformat()}] {role}: {content}\n"
            if metadata:
                trace_content += f"  Metadata: {json.dumps(metadata, indent=2)}\n"
            write_to_file(OUTPUT_TRACE_PATH, trace_content)
        except Exception as e:
            logger.error(f"Failed to write trace: {e}")

    def should_stop(self) -> tuple[bool, str]:
        """Determine if agent should stop execution."""
        if self.current_iteration >= self.max_iterations:
            return True, "Maximum iterations reached"
        
        if self.repetition_detector.is_repetitive():
            return True, "Repetitive behavior detected"
        
        if self.state == AgentState.ERROR:
            return True, "Error state reached"
        
        return False, ""

    def think(self) -> None:
        """Enhanced thinking process with better state management."""
        self.state = AgentState.THINKING
        self.current_iteration += 1
        self.metrics.iterations = self.current_iteration
        
        logger.info(f"Starting iteration {self.current_iteration}")
        self.trace("system", f"Iteration {self.current_iteration} started", {
            "iteration": self.current_iteration,
            "elapsed_time": self.metrics.elapsed_time
        })

        # Check stopping conditions
        should_stop, reason = self.should_stop()
        if should_stop:
            self.state = AgentState.MAX_ITERATIONS_REACHED if "iterations" in reason else AgentState.ERROR
            self.final_answer = f"Stopping execution: {reason}. Current progress: {self.memory.get_formatted_history()}"
            self.trace("assistant", self.final_answer)
            return

        # Generate prompt
        prompt = self.template.format(
            query=self.query,
            history=self.memory.get_formatted_history(),
            tools=', '.join([tool.metadata.description for tool in self.tools.values()]),
            iteration=self.current_iteration,
            max_iterations=self.max_iterations
        )

        # Get response from model
        response = self.ask_gemini(prompt)
        logger.info(f"Model response: {response}")
        
        self.trace("assistant", f"Thought: {response}", {
            "iteration": self.current_iteration,
            "response_length": len(response)
        })
        
        self.decide(response)

    def parse_response(self, response: str) -> ActionResponse:
        """Parse and validate model response."""
        try:
            # Clean response
            cleaned = response.strip().strip('`').strip()
            if cleaned.startswith('json'):
                cleaned = cleaned[4:].strip()
            
            # Parse JSON
            parsed = json.loads(cleaned)
            
            # Validate and create ActionResponse
            return ActionResponse(**parsed)
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Try to extract answer from raw text
            if "answer:" in response.lower():
                answer = response.split("answer:", 1)[1].strip()
                return ActionResponse(answer=answer, confidence=0.5)
            
            return ActionResponse(
                thought="I need to rethink this approach",
                confidence=0.0
            )
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return ActionResponse(
                thought="I encountered an error processing the response",
                confidence=0.0
            )

    def decide(self, response: str) -> None:
        """Enhanced decision making with structured parsing."""
        self.state = AgentState.DECIDING
        
        try:
            action_response = self.parse_response(response)
            
            if action_response.answer:
                # Final answer provided
                self.state = AgentState.COMPLETED
                self.final_answer = action_response.answer
                self.trace("assistant", f"Final Answer: {action_response.answer}", {
                    "confidence": action_response.confidence,
                    "total_iterations": self.current_iteration
                })
                
            elif action_response.action:
                # Action to perform
                action = action_response.action
                tool_name_str = action.get("name", "").upper()
                
                if tool_name_str == "NONE":
                    logger.info("No action needed, continuing to think")
                    self.think()
                else:
                    try:
                        tool_name = Name[tool_name_str]
                        query = action.get("input", self.query)
                        
                        self.trace("assistant", f"Action: Using {tool_name} tool", {
                            "tool": tool_name_str,
                            "query": query,
                            "confidence": action_response.confidence
                        })
                        
                        self.repetition_detector.add_action(f"{tool_name}:{query}")
                        self.act(tool_name, query)
                        
                    except KeyError:
                        logger.error(f"Unknown tool: {tool_name_str}")
                        self.trace("system", f"Error: Unknown tool {tool_name_str}")
                        self.think()
                        
            else:
                # Continue thinking
                self.think()
                
        except Exception as e:
            logger.error(f"Decision error: {e}")
            self.metrics.errors += 1
            self.state = AgentState.ERROR
            self.trace("system", f"Decision error: {str(e)}")
            self.think()

    def act(self, tool_name: Name, query: str) -> None:
        """Execute tool action with enhanced error handling."""
        self.state = AgentState.ACTING
        self.metrics.tool_calls += 1
        
        tool = self.tools.get(tool_name)
        if not tool:
            logger.error(f"Tool {tool_name} not registered")
            self.trace("system", f"Error: Tool {tool_name} not found")
            self.think()
            return
        
        try:
            start_time = time.time()
            result = tool.use(query)
            execution_time = time.time() - start_time
            
            observation = f"Observation from {tool_name}: {result}"
            self.trace("system", observation, {
                "tool": str(tool_name),
                "query": query,
                "execution_time": execution_time,
                "tool_call_count": tool.call_count
            })
            
            # Add observation to memory
            obs_message = Message(
                role="system",
                content=observation,
                metadata={"tool_result": True}
            )
            self.memory.add_message(obs_message)
            
            self.think()
            
        except Exception as e:
            logger.error(f"Action execution error: {e}")
            self.metrics.errors += 1
            self.trace("system", f"Action error: {str(e)}")
            self.think()

    def execute(self, query: str) -> str:
        """Execute query with enhanced monitoring."""
        self.state = AgentState.INITIALIZING
        self.query = query
        self.metrics = ExecutionMetrics(start_time=time.time())
        
        logger.info(f"Starting execution for query: {query}")
        self.trace("user", query, {"query_length": len(query)})
        
        try:
            self.think()
            
            while self.state not in [AgentState.COMPLETED, AgentState.ERROR, AgentState.MAX_ITERATIONS_REACHED]:
                time.sleep(0.1)  # Small delay to prevent busy waiting
            
            if self.final_answer:
                return self.final_answer
            elif self.memory.messages:
                return self.memory.messages[-1].content
            else:
                return "No response generated"
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            self.state = AgentState.ERROR
            return f"Execution failed: {str(e)}"
        finally:
            self.log_execution_summary()

    def log_execution_summary(self) -> None:
        """Log execution summary."""
        summary = {
            "total_iterations": self.metrics.iterations,
            "total_tool_calls": self.metrics.tool_calls,
            "total_errors": self.metrics.errors,
            "cache_hits": self.metrics.cache_hits,
            "execution_time": self.metrics.elapsed_time,
            "final_state": self.state.name
        }
        
        logger.info(f"Execution Summary: {json.dumps(summary, indent=2)}")
        self.trace("system", "Execution completed", summary)

    def ask_gemini(self, prompt: str) -> str:
        """Query Gemini with enhanced error handling."""
        try:
            contents = [Part.from_text(prompt)]
            response = generate(self.model, contents)
            return str(response) if response is not None else "No response from Gemini"
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error communicating with Gemini: {str(e)}"

# Flask app setup
app = Flask(__name__)
CORS(app)
def create_agent():
    logger.info("Creating agent...")
    gemini = GenerativeModel(config.MODEL_NAME)
    agent = Agent(model=gemini, config={
        "max_iterations": 15,
        "confidence_threshold": 0.7,
        "cache_ttl": 600,
        "max_history_size": 25
    })
    logger.info("Registering tools...")
    wiki_metadata = ToolMetadata(
        name=Name.WIKIPEDIA,
        description="Wikipedia search for factual information and biographies",
        usage_examples=["Barack Obama", "Python programming", "World War II"],
        max_retries=3
    )
    agent.register_tool(wiki_metadata, wiki_search)
    
    google_metadata = ToolMetadata(
        name=Name.GOOGLE,
        description="Google search for current information and web results",
        usage_examples=["current weather", "latest news", "restaurant reviews"],
        max_retries=3
    )
    agent.register_tool(google_metadata, google_search)

    logger.info(f"Registered tools: {list(agent.tools.keys())}")
    
    return agent

agent_instance = create_agent()

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        response = agent_instance.execute(query)
        
        return jsonify({"response": response, "status": "success"})
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    """Get list of conversations."""
    try:
        conversations = agent_instance.memory.get_recent_history()
        return jsonify([{"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()} for msg in conversations])
    except Exception as e:
        logger.error(f"Error fetching conversations: {e}")
        return jsonify({"error": str(e)}), 500
    


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)