import streamlit as st
import logging
import traceback
import os
import json
from typing import Tuple, Optional, Dict

# Import AutoGen components directly in app.py
import autogen
from autogen import UserProxyAgent, GroupChatManager, GroupChat, Agent

# Import necessary components from local modules
from LLMConfiguration import LLMConfiguration, VERTEX_AI, AZURE, ANTHROPIC
from common_functions import (
    create_agent,         # Still needed
    create_groupchat,     # Still needed
    create_groupchat_manager, # Still needed
    initiate_chat_task, # Import from common_functions
    run_agent_step,     # Import from common_functions
    send_user_message,   # Import from common_functions
    read_system_message # Import the function from common_functions
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---

PRODUCT_LEAD_NAME = "ProductLead"
PERSONA1_NAME = "Persona1"  # Updated name
PERSONA2_NAME = "Persona2"  # Updated name

PRODUCT_LEAD_SYS_MSG_FILE = "ProductLead.md"
# These files provide the *default* system messages for the personas
PERSONA1_DEFAULT_SYS_MSG_FILE = "PolicyGuard.md"
PERSONA2_DEFAULT_SYS_MSG_FILE = "Challenger.md"

# Marker for policy injection (if Persona1 takes on this role)
POLICY_INJECTION_MARKER = "## Policies"

MAX_MESSAGES_DISPLAY = 50 # Limit messages displayed to prevent clutter
POLICY_TEXT_KEY = "policy_text_input" # Key for the policy text area
TASK_PROMPT_KEY = "initial_prompt_input" # Key for the task description area

# Keys for editable system messages in session_state
PERSONA1_EDIT_KEY = "persona1_editable_prompt"
PERSONA2_EDIT_KEY = "persona2_editable_prompt"
PRODUCT_LEAD_EDIT_KEY = "product_lead_editable_prompt"

AGENT_CONFIG = {
    PERSONA1_NAME: {"file": PERSONA1_DEFAULT_SYS_MSG_FILE, "key": PERSONA1_EDIT_KEY},
    PERSONA2_NAME: {"file": PERSONA2_DEFAULT_SYS_MSG_FILE, "key": PERSONA2_EDIT_KEY},
    PRODUCT_LEAD_NAME: {"file": PRODUCT_LEAD_SYS_MSG_FILE, "key": PRODUCT_LEAD_EDIT_KEY},
}

# Feature 4: Context Limit for Gemini Pro 1.5 (approximate)
CONTEXT_LIMIT = 1_000_000
WARNING_THRESHOLD = 0.85 # Warn at 85% of context limit

# --- Helper Functions ---

def estimate_tokens(text: str) -> int:
    """Approximates token count using character count / 4."""
    return len(text or "") // 4

def initialize_editable_prompts():
    """Loads default agent prompts into session state if they don't exist."""
    for agent_name, config in AGENT_CONFIG.items():
        if config["key"] not in st.session_state:
            try:
                st.session_state[config["key"]] = read_system_message(config["file"])
                logger.info(f"Loaded default system message for {agent_name} into session state ({config['key']}).")
            except Exception as e:
                st.session_state[config["key"]] = f"Error loading default from {config['file']}: {e}"
                logger.error(f"Failed to load initial prompt for {agent_name} from {config['file']}: {e}")

def update_token_warning():
    """Calculates estimated total tokens and displays warning if near limit."""
    policy_text = st.session_state.get(POLICY_TEXT_KEY, "")
    task_text = st.session_state.get(TASK_PROMPT_KEY, "")
    persona1_prompt = st.session_state.get(PERSONA1_EDIT_KEY, "")
    persona2_prompt = st.session_state.get(PERSONA2_EDIT_KEY, "")
    product_lead_prompt = st.session_state.get(PRODUCT_LEAD_EDIT_KEY, "")

    policy_tokens = estimate_tokens(policy_text)
    task_tokens = estimate_tokens(task_text)
    persona1_tokens = estimate_tokens(persona1_prompt)
    persona2_tokens = estimate_tokens(persona2_prompt)
    product_lead_tokens = estimate_tokens(product_lead_prompt)

    total_system_prompt_tokens = persona1_tokens + persona2_tokens + product_lead_tokens
    total_input_tokens = policy_tokens + task_tokens # Policy text is separate input
    total_estimated_tokens = total_input_tokens + total_system_prompt_tokens

    if 'token_info_placeholder' in globals():
        token_info_placeholder.caption(f"Estimated Input Tokens: ~{total_estimated_tokens:,} / {CONTEXT_LIMIT:,}")

    if 'token_warning_placeholder' in globals():
        if total_estimated_tokens > CONTEXT_LIMIT * WARNING_THRESHOLD:
            token_warning_placeholder.warning(f"Inputs approaching context limit ({WARNING_THRESHOLD*100:.0f}%). Total: ~{total_estimated_tokens:,}")
        else:
            token_warning_placeholder.empty()

def setup_chat(
    llm_provider: str = VERTEX_AI,
    model_name: str = "gemini-1.5-pro-002",
    policy_text: Optional[str] = None, # Policy text to be injected (optional)
    agent_prompts: Dict[str, str] = {}
    ) -> Tuple[GroupChatManager, UserProxyAgent]:
    """
    Sets up the agents, group chat, and manager.
    Uses provided agent prompts from session state.
    Injects provided policy_text into Persona1's system message if policy_text is given.
    """
    logger.info(f"Setting up chat with provider: {llm_provider}, model: {model_name}")
    if policy_text:
        logger.info("Policy text provided, will inject into Persona1 if marker found.")
    else:
        logger.info("No policy text provided for injection.")

    # --- LLM Configuration ---

    if llm_provider == VERTEX_AI:
        try:
            if "gcp_credentials" not in st.secrets:
                 raise ValueError("Missing 'gcp_credentials' section in Streamlit secrets.")
            required_keys = ["project_id", "private_key", "client_email", "type"]
            if not all(key in st.secrets["gcp_credentials"] for key in required_keys):
                 raise ValueError(f"Missing required keys ({required_keys}) within 'gcp_credentials' in Streamlit secrets.")
            vertex_credentials = dict(st.secrets["gcp_credentials"])
            llm_config = LLMConfiguration(
                VERTEX_AI,
                model_name,
                project_id=vertex_credentials.get('project_id'),
                location="us-central1",
                vertex_credentials=vertex_credentials,
            )
            logger.info("Vertex AI LLM configuration loaded from st.secrets.")
        except ValueError as e:
             logger.error(f"Credential error: {e}")
             raise
        except Exception as e:
            logger.error(f"Error loading Vertex AI credentials from st.secrets: {e}", exc_info=True)
            raise
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    if not llm_config.get_config():
        raise ValueError("Failed to create a valid LLM configuration object or dictionary.")

    # --- Agent Creation (Using Session State Prompts) ---

    try:
        agents = {}
        for agent_name, config in AGENT_CONFIG.items():
            system_message_content = agent_prompts.get(agent_name)
            if not system_message_content:
                 logger.error(f"Missing system message content for agent {agent_name} in setup_chat call.")
                 try:
                     system_message_content = read_system_message(config["file"]) # Fallback to default
                     logger.warning(f"Had to re-read {config['file']} for {agent_name} during setup.")
                 except Exception as e:
                     raise ValueError(f"Could not load system message for {agent_name}: {e}")

            # Inject policy into Persona1's message if policy_text is provided
            if agent_name == PERSONA1_NAME and policy_text and policy_text.strip():
                 base_persona1_sys_msg = system_message_content
                 if POLICY_INJECTION_MARKER in base_persona1_sys_msg:
                     system_message_content = base_persona1_sys_msg.replace(
                         POLICY_INJECTION_MARKER,
                         f'{POLICY_INJECTION_MARKER}

{policy_text.strip()}',
                         1
                     )
                     logger.info(f"Injected policy text into {PERSONA1_NAME} system message under '{POLICY_INJECTION_MARKER}'.")
                 else:
                     logger.warning(f"Policy injection marker '{POLICY_INJECTION_MARKER}' not found in {PERSONA1_NAME} prompt. Appending policy text instead.")
                     system_message_content += f"

## Policies

{policy_text.strip()}" # Ensure it has a header

            agent_type = "user_proxy" if agent_name == PRODUCT_LEAD_NAME else "assistant"
            agents[agent_name] = create_agent(
                name=agent_name,
                llm_config=llm_config,
                system_message_content=system_message_content,
                system_message_file=None, # File content is already loaded
                agent_type=agent_type,
            )

        product_lead_agent = agents[PRODUCT_LEAD_NAME]
        persona1_agent = agents[PERSONA1_NAME]
        persona2_agent = agents[PERSONA2_NAME]

        logger.info("Agents created successfully using session state prompts.")
    except Exception as e:
        logger.error(f"Unexpected error during agent creation: {e}", exc_info=True)
        raise

    # --- Group Chat Setup ---

    # Order might matter for turn-taking, ensure ProductLead is correctly placed
    # For a group chat where user (ProductLead) talks to Persona1 and Persona2,
    # they should all be in the list.
    chat_participants = [product_lead_agent, persona1_agent, persona2_agent]
    try:
        groupchat = create_groupchat(chat_participants, max_round=50) # Increased max_round
        logger.info("GroupChat created successfully.")
    except Exception as e:
        logger.error(f"Unexpected error during GroupChat creation: {e}", exc_info=True)
        raise

    manager_llm_config = llm_config # Reusing the same config for the manager
    try:
        manager = create_groupchat_manager(groupchat, manager_llm_config)
        logger.info("GroupChatManager created successfully.")
    except Exception as e:
        logger.error(f"Unexpected error during GroupChatManager creation: {e}", exc_info=True)
        raise

    logger.info("Chat setup completed.")
    return manager, product_lead_agent # Return the user_proxy_agent

# --- Streamlit App UI ---

st.title("🤖 AI Persona Chat Session")

# --- Initialization ---

default_values = {
    "chat_initialized": False,
    "processing": False,
    "error_message": None,
    "config": None,
    "manager": None,
    "product_lead_agent": None, # This is the UserProxyAgent
    "messages": [],
    "next_agent": None,
    TASK_PROMPT_KEY: "",
    POLICY_TEXT_KEY: "", # For optional policy injection into Persona1
}
for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

initialize_editable_prompts()

# --- Configuration Loading (Placeholder) ---

if not st.session_state.config:
    try:
        # Simplified: Using default config. Replace with actual load_config if needed.
        st.session_state.config = {
            "llm_provider": VERTEX_AI,
            "model_name": "gemini-1.5-pro-002" # Default model
        }
        # st.sidebar.success("Configuration loaded (using defaults).") # Less verbose
    except Exception as e:
        st.sidebar.error(f"Failed to load configuration: {e}")
        st.stop()

# --- Agent Configuration Expander (Sidebar) ---

with st.sidebar.expander("Configure AI Personas & Task", expanded=True):
    st.caption("Define the AI personas and the initial task.")

    # Editable system prompts for AI Personas
    for agent_name, config_info in AGENT_CONFIG.items():
        # Exclude ProductLead from this editable section if its prompt is fixed or managed differently
        if agent_name == PRODUCT_LEAD_NAME: # ProductLead prompt might not be user-editable in the same way
            # If ProductLead prompt is also editable, include it. Otherwise, skip.
            # For now, let's assume ProductLead's system message is less frequently changed or fixed.
            # If you want ProductLead to be editable here, remove this conditional block.
            # st.text_area(
            #     f"Edit {agent_name} System Prompt",
            #     key=config_info["key"],
            #     height=100,
            #     disabled=st.session_state.chat_initialized,
            #     on_change=update_token_warning
            # )
            pass # Skip ProductLead here, assuming its role is more fixed as the user proxy.
        else:
            st.text_area(
                f"System Prompt for {agent_name} (User Defined Name):",
                key=config_info["key"], # e.g., persona1_editable_prompt
                height=150,
                disabled=st.session_state.chat_initialized,
                help=f"Define the background, personality, and instructions for {agent_name}. The actual name used in chat will be {agent_name}.",
                on_change=update_token_warning
            )

    # Token information and warning placeholders
    token_info_placeholder = st.sidebar.empty()
    token_warning_placeholder = st.sidebar.empty()

    # Optional Policy Text (for Persona1)
    policy_text_input = st.sidebar.text_area(
        "Enter Policy Text (Optional, for Persona1):",
        height=100,
        key=POLICY_TEXT_KEY,
        disabled=st.session_state.chat_initialized,
        help="If provided, this text will be injected into Persona1's system message under '## Policies'.",
        on_change=update_token_warning
    )

    # Task Description
    initial_prompt_input = st.sidebar.text_area(
        "Enter the Initial Task or Question:",
        height=150,
        key=TASK_PROMPT_KEY,
        disabled=st.session_state.chat_initialized,
        help="Describe the initial task, question, or topic for the discussion.",
        on_change=update_token_warning
    )
    update_token_warning() # Initial call

# --- Start Chat Area ---

if st.sidebar.button("🚀 Start Chat", key="start_chat",
                    disabled=st.session_state.chat_initialized
                             or not st.session_state.get(TASK_PROMPT_KEY, "").strip() # Ensure task is not empty
                             or not st.session_state.config):

    task_prompt = st.session_state.get(TASK_PROMPT_KEY, "").strip()
    if not st.session_state.chat_initialized and task_prompt and st.session_state.config:
        st.session_state.processing = True
        st.session_state.error_message = None
        try:
            with st.spinner("Brewing conversations... Setting up AI personas..."):
                logger.info("Setting up chat components with user-defined persona prompts and task...")

                current_agent_prompts = {}
                for agent_name, config_info in AGENT_CONFIG.items():
                    current_agent_prompts[agent_name] = st.session_state.get(config_info["key"], f"Error: Missing prompt for {agent_name}")

                st.session_state.manager, st.session_state.product_lead_agent = setup_chat(
                    llm_provider=st.session_state.config.get("llm_provider", VERTEX_AI),
                    model_name=st.session_state.config.get("model_name", "gemini-1.5-pro-002"),
                    policy_text=st.session_state.get(POLICY_TEXT_KEY, ""),
                    agent_prompts=current_agent_prompts
                )
                logger.info("Setup complete. Initiating chat task...")

                # ProductLead (UserProxyAgent) initiates the chat with the task prompt
                initial_messages, next_agent = initiate_chat_task(
                    st.session_state.product_lead_agent, # User proxy agent
                    st.session_state.manager,          # Group chat manager
                    task_prompt                        # The initial message / task
                )
                st.session_state.messages = initial_messages
                st.session_state.next_agent = next_agent # Should be one of the AI personas or manager choice
                st.session_state.chat_initialized = True
                logger.info(f"Chat initiated by {PRODUCT_LEAD_NAME}. Task: '{task_prompt}'. Next agent: {st.session_state.next_agent.name if st.session_state.next_agent else 'None'}")

        except Exception as e:
            logger.error(f"Error setting up or initiating chat task: {traceback.format_exc()}")
            st.session_state.error_message = f"Setup/Initiation failed: {e}"
            st.session_state.chat_initialized = False
            st.session_state.manager = None
            st.session_state.product_lead_agent = None
        finally:
            st.session_state.processing = False
        st.rerun()

# --- Display Error Message ---

if st.session_state.error_message:
    st.error(st.session_state.error_message)

# --- Main Chat Interaction Area ---

chat_container = st.container()

with chat_container:
    if st.session_state.chat_initialized and st.session_state.manager:
        display_messages(st.session_state.messages) # Existing function to display messages

        if st.session_state.next_agent and not st.session_state.processing:
            next_agent_name = st.session_state.next_agent.name

            # Check if it's the user's (ProductLead's) turn
            if next_agent_name == PRODUCT_LEAD_NAME:
                st.markdown(f"**Your turn (as User):**") # Simplified user role display
                form_key = f'user_input_form_{len(st.session_state.messages)}'
                input_key = f"user_input_{len(st.session_state.messages)}"

                with st.form(key=form_key):
                    user_input = st.text_input(
                        "Enter your message:",
                        key=input_key,
                        disabled=st.session_state.processing,
                        placeholder="Type your message and press Enter or click Send..."
                    )
                    submitted = st.form_submit_button(
                        "✉️ Send Message",
                        disabled=st.session_state.processing
                    )
                    if submitted:
                        if not user_input.strip():
                             st.warning("Please enter a message.")
                        else:
                            st.session_state.processing = True
                            st.session_state.error_message = None
                            should_rerun = False
                            with st.spinner(f"Sending message..."):
                                try:
                                    logger.info(f"Sending user message: {user_input}")
                                    # UserProxyAgent (product_lead_agent) sends the message to the manager
                                    new_messages, next_agent = send_user_message(
                                        st.session_state.manager,
                                        st.session_state.product_lead_agent, # The agent sending the message
                                        user_input
                                    )
                                    st.session_state.messages.extend(new_messages)
                                    st.session_state.next_agent = next_agent
                                    logger.info(f"User message sent. Next agent: {next_agent.name if next_agent else 'None'}")
                                    should_rerun = True
                                except Exception as e:
                                     logger.error(f"Error sending user message: {traceback.format_exc()}")
                                     st.session_state.error_message = f"Error sending message: {e}"
                                     should_rerun = True
                            st.session_state.processing = False
                            if should_rerun:
                                st.rerun()
            else: # Auto-run AI Persona's Turn
                st.markdown(f"**Running turn for:** {next_agent_name}...")
                st.session_state.processing = True
                st.session_state.error_message = None
                should_rerun = False
                with st.spinner(f"Thinking... {next_agent_name} is responding..."):
                    try:
                        logger.info(f"Running step for AI agent: {next_agent_name}")
                        new_messages, next_agent = run_agent_step(
                            st.session_state.manager,
                            st.session_state.next_agent # The AI agent whose turn it is
                        )
                        st.session_state.messages.extend(new_messages)
                        st.session_state.next_agent = next_agent
                        logger.info(f"AI Agent {next_agent_name} finished. Next agent: {next_agent.name if next_agent else 'None'}")
                        should_rerun = True
                    except Exception as e:
                        logger.error(f"Error during {next_agent_name}'s turn: {traceback.format_exc()}")
                        st.session_state.error_message = f"Error during {next_agent_name}'s turn: {e}"
                        st.session_state.next_agent = None # Stop chat on agent error
                        should_rerun = True
                st.session_state.processing = False
                if should_rerun:
                     st.rerun()
        elif not st.session_state.next_agent and st.session_state.chat_initialized and not st.session_state.processing:
             st.success("Chat finished or awaiting next step.")

# --- Clear Chat Button ---

if st.session_state.chat_initialized or st.session_state.error_message or not st.session_state.manager:
     if st.sidebar.button("Clear Chat / Reset", key="clear_chat"):
         st.session_state.chat_initialized = False
         st.session_state.processing = False
         st.session_state.error_message = None
         st.session_state.messages = []
         st.session_state.next_agent = None
         st.session_state.manager = None
         st.session_state.product_lead_agent = None
         st.session_state[TASK_PROMPT_KEY] = ""
         st.session_state[POLICY_TEXT_KEY] = ""
         # Optionally reset editable prompts to default (currently keeps user edits)
         # initialize_editable_prompts() # Uncomment to reset persona prompts to default file content on clear
         logger.info("Chat state cleared. Ready for new configuration/start.")
         update_token_warning()
         st.rerun()

def display_messages(messages):
    """Displays chat messages, limiting the number shown."""
    num_messages = len(messages)
    start_index = max(0, num_messages - MAX_MESSAGES_DISPLAY)
    if num_messages > MAX_MESSAGES_DISPLAY:
        st.warning(f"Displaying last {MAX_MESSAGES_DISPLAY} of {num_messages} messages.")

    for i, msg in enumerate(messages[start_index:], start=start_index):
        sender_name = msg.get("name", "System")
        # If name is not present but role is, use role (e.g. for system messages from manager)
        if not sender_name and "role" in msg:
             sender_name = msg["role"].capitalize()
        # If sender is "User" use ProductLead's name, otherwise use the agent's name or "System"
        # Autogen typically sets 'name' for agents, and 'role': 'user' for UserProxyAgent messages.
        # The 'name' field in the message dictionary is the most reliable source for the sender.

        content = msg.get("content", "")
        if isinstance(content, list): # Handle cases like tool calls or multi-part messages
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item: parts.append(item["text"])
                elif isinstance(item, str): parts.append(item)
                else: parts.append(str(item)) # Fallback for other types
            content = "
".join(parts)
        elif not isinstance(content, str):
             content = str(content) # Ensure content is string

        # Determine avatar and alignment based on sender
        # ProductLead (user's actual agent) will have its messages shown as "User"
        if sender_name == PRODUCT_LEAD_NAME: #This is our UserProxyAgent
            with st.chat_message("user", avatar="🧑"): # Display as 'user' for UI
                 st.markdown(f"""**You ({PRODUCT_LEAD_NAME}):**
{content}""")
        elif sender_name in [PERSONA1_NAME, PERSONA2_NAME]: # AI personas
            with st.chat_message("assistant", avatar="🤖"):
                 st.markdown(f"""**{sender_name}:**
{content}""")
        else: # System messages or other agents (e.g. GroupChatManager talking)
            with st.chat_message("system", avatar="⚙️"): # Or a generic avatar
                 st.markdown(f"""_{sender_name}: {content}_""")

