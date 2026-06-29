import os
import sys

# --- CONFIGURATION & ENV LOADING ---
# Define default configuration values
config = {
    "BOT_NAME": "RuleBot",
    "DEFAULT_GREETING": "Hello! I am RuleBot, a deterministic rule-based AI chatbot. How can I help you today?",
    "DEBUG_MODE": "True"
}

def load_env(env_path=".env"):
    """
    Loads environment variables from a .env file.
    Implements a custom fallback parser if python-dotenv is not installed,
    ensuring zero-dependency execution.
    """
    # 1. Attempt to use python-dotenv if installed
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        pass  # Fall back to custom parser below
    
    # 2. Parse manually (robust implementation)
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Split at first '='
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    # Strip wrapping quotes if any
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

# Load the environment file
load_env()

# Update configurations from environment variables
BOT_NAME = os.environ.get("BOT_NAME", config["BOT_NAME"])
DEFAULT_GREETING = os.environ.get("DEFAULT_GREETING", config["DEFAULT_GREETING"])
DEBUG_MODE = os.environ.get("DEBUG_MODE", config["DEBUG_MODE"]).strip().lower() in ("true", "1", "yes")

# --- KNOWLEDGE BASE ---
# Predefined user queries mapped to responses.
# Demonstrates practical experience with dictionaries.
KNOWLEDGE_BASE = {
    "greetings": {
        "patterns": ["hello", "hi", "hey", "greetings", "hola", "yo"],
        "response": f"Hi there! I am {BOT_NAME}. How can I help you today?"
    },
    "identity": {
        "patterns": ["name", "who are you", "what is your name", "tell me about yourself"],
        "response": f"I am {BOT_NAME}, a Rule-Based AI Chatbot. I make decisions using deterministic rules!"
    },
    "status": {
        "patterns": ["how are you", "how's it going", "how are you doing", "how do you do"],
        "response": "I'm running smoothly on your computer, thank you for asking! How are you?"
    },
    "weather": {
        "patterns": ["weather", "forecast", "temperature", "rain", "sun"],
        "response": "I don't have access to live weather sensors, but I hope you are having a pleasant day!"
    },
    "help": {
        "patterns": ["help", "what can you do", "commands", "menu"],
        "response": (
            "I can answer basic questions! Try asking about my name, weather, how I am, or say hello.\n"
            "Special commands:\n"
            "  /ipo   - Explain the Input-Process-Output model of this chatbot\n"
            "  /debug - Toggle verbose IPO visual debugging\n"
            "  /exit  - Close the conversation (or type 'exit', 'quit', 'bye')"
        )
    },
    "exit": {
        "patterns": ["exit", "quit", "bye", "goodbye", "close"],
        "response": "Goodbye! Thank you for chatting. Have a great day!"
    }
}

# --- IPO MODEL IMPLEMENTATION ---

def explain_ipo():
    """
    Returns an educational explanation of the Input -> Process -> Output model.
    """
    return (
        f"\n=================== THE IPO MODEL IN {BOT_NAME.upper()} ===================\n"
        "1. INPUT: The user types a message in the terminal. The raw string is received.\n"
        "2. PROCESS: The chatbot normalizes the text (converts to lowercase, removes outer whitespace),\n"
        "   checks for exit commands, and runs a pattern matching algorithm (deterministic rule-based logic)\n"
        "   against a predefined dictionary.\n"
        "3. OUTPUT: The chatbot selects the matched response and prints it to the terminal screen.\n"
        "===================================================================\n"
    )

def get_response(raw_input):
    """
    Processes the raw user input and matches it against the knowledge base.
    Demonstrates:
    - Input Normalization (lowercase, stripping whitespace)
    - Deterministic Decision Making (rule-based pattern matching)
    - Output Selection (dictionary lookup & fallback logic)
    """
    # 1. INPUT STAGE: The input is already received via the parameter `raw_input`
    
    # 2. PROCESS STAGE:
    # A. Normalization (Clean the raw input)
    normalized_input = raw_input.lower().strip()
    
    # If the user input is empty after stripping
    if not normalized_input:
        return normalized_input, "empty_input", "I noticed you didn't say anything! Feel free to type something or ask for 'help'."
    
    # Special CLI commands
    if normalized_input == "/ipo":
        return normalized_input, "explain_ipo_command", explain_ipo()
    
    # B. Rule Matching Logic
    # Loop through our dictionary keys (categories) to find a pattern match
    matched_category = None
    for category, content in KNOWLEDGE_BASE.items():
        # Check if any defined pattern is in the normalized user input
        for pattern in content["patterns"]:
            if pattern in normalized_input:
                matched_category = category
                break
        if matched_category:
            break
            
    # C. Output selection
    if matched_category:
        response = KNOWLEDGE_BASE[matched_category]["response"]
    else:
        # Fallback mechanism for unrecognized queries
        response = "I'm sorry, I didn't quite understand that. Could you rephrase? (Type 'help' for options)"
        
    # 3. OUTPUT STAGE: Return the selected response
    return normalized_input, matched_category, response

def main():
    global DEBUG_MODE
    print(f"--- Welcome to the Rule-Based AI Chatbot ({BOT_NAME}) ---")
    print(f"Initial Greeting: {DEFAULT_GREETING}")
    print("Type 'help' to see what I can do, '/ipo' to see the IPO model, or '/exit' to quit.\n")
    
    # Continuous loop keeps the chatbot running until explicit exit command
    while True:
        try:
            # 1. INPUT (User enters query)
            raw_input = input("You: ")
            
            # Normalization check for exit commands first
            check_exit = raw_input.lower().strip()
            if check_exit in ["/exit", "exit", "quit", "bye", "goodbye"]:
                print(f"{BOT_NAME}: {KNOWLEDGE_BASE['exit']['response']}")
                break
            
            # Interactive toggle for debug mode
            if check_exit == "/debug":
                DEBUG_MODE = not DEBUG_MODE
                print(f"System: DEBUG_MODE set to {DEBUG_MODE}")
                continue
            
            # 2. PROCESS & 3. OUTPUT
            normalized, category, response = get_response(raw_input)
            
            # Print verbose IPO logs if DEBUG_MODE is True
            if DEBUG_MODE:
                print("\n[IPO PIPELINE DIAGNOSTIC]")
                print(f"  [INPUT]   Raw Input:   '{raw_input}'")
                print(f"  [PROCESS] Sanitized:   '{normalized}'")
                print(f"            Decision:    Rule-based dictionary search")
                print(f"            Matched Cat: {category if category else 'None (Fallback triggered)'}")
                print(f"  [OUTPUT]  Response:    '{response}'\n")
            
            print(f"{BOT_NAME}: {response}")
            
        except (KeyboardInterrupt, EOFError):
            # Graceful exit on Ctrl+C or EOF (Ctrl+Z/D)
            print(f"\n{BOT_NAME}: {KNOWLEDGE_BASE['exit']['response']}")
            sys.exit(0)

if __name__ == "__main__":
    main()
