import logging
# import time # Unused import removed
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union
import json # Import json for pretty printing dicts
import re # Import re for regular expressions

from LLMConfiguration import LLMConfiguration, logger # Assuming logger is correctly configured here

# Configure logging (if not already done elsewhere)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # User should ensure DEBUG level is set in active config


def read_system_message(filename: str) -> Tuple[Optional[str], str]:
    """
    Reads system message from a file.
    Parses a DisplayName from the first line if present.
    Returns a tuple: (display_name, system_message_content).
    display_name will be None if not found.
    system_message_content will be the original content if display_name is not found,
    or the content without the DisplayName line.
    Returns (None, "You are a helpful assistant.") on error or if file not found.
    """
    default_system_message = "You are a helpful assistant."
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            full_content = file.read()
            lines = full_content.splitlines()
            display_name = None
            system_message_content = full_content.strip()

            if lines:
                # Check for DisplayName: Your Name format
                match = re.match(r"^DisplayName:\s*(.+)$", lines[0].strip(), re.IGNORECASE)
                if match:
                    display_name = match.group(1).strip()
                    # Join the rest of the lines for the system message
                    system_message_content = "
".join(lines[1:]).strip()
                    logger.info(f"Parsed DisplayName '{display_name}' from {filename}")
                else:
                    logger.info(f"No DisplayName found in first line of {filename}. Using full content for system message.")
            else: # Empty file
                 logger.warning(f"System message file {filename} is empty. Using default system message.")
                 return None, default_system_message


            return display_name, system_message_content

    except FileNotFoundError:
        logger.error(f"System message file not found: {filename}")
        return None, default_system_message
    except Exception as e:
        logger.error(f"Error reading system message file {filename}: {e}")
        return None, default_system_message


def create_agent(
    name: str,
    llm_config: LLMConfiguration,
    system_message_file: Optional[str] = None,
    system_message_content: Optional[str] = None, # This will be the content *without* DisplayName
    agent_type="assistant"
    ) -> autogen.Agent:
    """
    Creates an agent with the specified configuration.
    Prioritizes system_message_content if provided.
    If system_message_file is used, it implies DisplayName has already been handled.
    """
    system_message_for_agent = None # This is what the agent actually uses
    # display_name_for_agent = name # Default display name is the agent's internal name

    if system_message_content:
        system_message_for_agent = system_message_content.strip()
        logger.debug(f"Using provided system message content for agent {name}.")
    elif system_message_file:
        # If only file is given, read_system_message (which parses DisplayName)
        # should ideally be called *before* this function.
        # For now, assume if file is passed here, it's pre-processed or doesn't have DisplayName.
        # This part of the logic might need refinement depending on call patterns in app.py.
        # For safety, we call our modified read_system_message here if content isn't directly provided.
        # However, app.py will be modified to call read_system_message first.
        _ , system_msg_from_file = read_system_message(system_message_file)
        system_message_for_agent = _system_msg_from_file
        logger.debug(f"Read system message from file {system_message_file} for agent {name}. DisplayName parsing handled separately.")
    else:
        logger.error(f"Neither system_message_file nor system_message_content provided for agent {name}. Using default.")
        system_message_for_agent = "You are a helpful assistant."

    if not system_message_for_agent:
         logger.error(f"Failed to obtain a system message for agent {name}. Cannot create agent.")
         raise ValueError(f"System message is empty or could not be loaded for agent {name}")

    config = llm_config.get_config()
    if not config:
         logger.error(f"LLM configuration invalid or missing for agent {name}. Cannot create agent.")
         raise ValueError(f"Invalid LLM configuration for agent {name}")

    if agent_type == "user_proxy":
        agent = UserProxyAgent(
            name=name,
            system_message=system_message_for_agent,
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=config,
            default_auto_reply="",
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        )
    else:  # AssistantAgent
        agent = AssistantAgent(
            name=name,
            system_message=system_message_for_agent,
            human_input_mode="NEVER",
            llm_config=config,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            )
    logger.info(f"Agent '{name}' created successfully (Type: {agent_type}).")
    return agent


# --- Updated Custom Speaker Selection Logic -- PRIORITIZE LAST MENTION ---
def custom_speaker_selection(last_speaker: Agent, groupchat: GroupChat) -> Agent:
    """
    Selects the next speaker based on the *last* mention in the last message.
    Defaults to ProductLead if no specific mention is found or on error.
    """
    logger.debug(f"--- Entering custom_speaker_selection ---")
    logger.debug(f"Last speaker: {last_speaker.name if last_speaker else 'None'}")
    logger.debug(f"Available agents: {[a.name for a in groupchat.agents]}")

    product_lead_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not product_lead_agent:
        logger.error("No UserProxyAgent (ProductLead) found in custom_speaker_selection!")
        logger.debug("Selecting first agent as fallback (ProductLead not found).")
        # Ensure groupchat.agents is not empty before accessing index 0
        if not groupchat.agents:
             logger.critical("Groupchat has no agents!")
             raise ValueError("Cannot select speaker from empty agent list.")
        return groupchat.agents[0]

    if not groupchat.messages:
        logger.info("No messages yet, selecting ProductLead by default.")
        logger.debug(f"--- Exiting custom_speaker_selection (No messages) ---")
        return product_lead_agent

    last_message = None
    message_content = ''
    try:
        last_message = groupchat.messages[-1]
        try:
            logger.debug(f"Last message received by custom_speaker_selection: {json.dumps(last_message)}")
        except Exception as e:
             logger.warning(f"Could not json.dumps last_message for logging in custom_speaker_selection: {e}")
             logger.debug(f"Last message (raw) received by custom_speaker_selection: {last_message}")

        message_content = last_message.get('content', '')
        if not isinstance(message_content, str):
            logger.warning(f"Message content was not a string ({type(message_content)}), converting to string.")
            message_content = str(message_content)

        logger.debug(f"Extracted message content for selection check: '{message_content[:100]}...'" + ('...' if len(message_content) > 100 else ''))

        if message_content.rstrip().endswith("TERMINATE"):
            logger.info("Termination message detected. Selecting ProductLead for final step.")
            logger.debug(f"--- Exiting custom_speaker_selection (Terminate) ---")
            return product_lead_agent

    except (AttributeError, IndexError, KeyError) as e:
        logger.warning(f"Error accessing last message content ({e}), defaulting to ProductLead")
        logger.debug(f"--- Exiting custom_speaker_selection (Error accessing message) ---")
        return product_lead_agent

    # --- Agent Mention Logic (Prioritize Last Mention) ---
    agent_patterns = {
        "PolicyGuard": ["PolicyGuard"],
        "Challenger": ["Challenger"], # Updated name
        "ProductLead": ["ProductLead"]
    }

    lower_message_content = message_content.lower()
    last_mention_index = -1
    agent_to_select = None

    # Find the agent mentioned *last* in the message
    for agent in groupchat.agents:
        if agent.name in agent_patterns:
            patterns = agent_patterns[agent.name]
            for pattern in patterns:
                # Find the last occurrence of the pattern (agent name)
                try:
                    current_index = lower_message_content.rindex(pattern.lower())
                    if current_index > last_mention_index:
                        # Found a mention later in the string
                        last_mention_index = current_index
                        agent_to_select = agent
                        logger.debug(f"Found potential mention of '{agent.name}' at index {current_index}. Currently selected.")
                    elif current_index == last_mention_index:
                        # Tie-breaking: if two mentions end at the same spot (unlikely with names), keep existing
                        logger.debug(f"Found mention of '{agent.name}' at same index {current_index}, keeping previous selection '{agent_to_select.name if agent_to_select else 'None'}'.")
                except ValueError:
                    # Substring not found, continue to next pattern/agent
                    continue

    # --- Selection based on Last Mention ---
    next_speaker = None
    if agent_to_select and agent_to_select != last_speaker:
        # An agent was mentioned last, and it's not the speaker themselves
        next_speaker = agent_to_select
        logger.info(f"Last mention match found for '{next_speaker.name}'. Selecting.")
        logger.debug(f"--- Exiting custom_speaker_selection (Last Mention found) ---")
        return next_speaker
    elif agent_to_select and agent_to_select == last_speaker:
         # The only agent mentioned last was the speaker themself. Default to ProductLead.
         logger.info(f"Last mentioned agent was the speaker ({last_speaker.name}). Defaulting to ProductLead.")
         logger.debug(f"--- Exiting custom_speaker_selection (Self-mention default) ---")
         return product_lead_agent
    else:
        # No mentions found at all. Default to ProductLead.
        logger.info(f"No specific next agent mentioned. Defaulting to ProductLead.")
        logger.debug(f"--- Exiting custom_speaker_selection (Defaulting to ProductLead) ---")
        return product_lead_agent


def create_groupchat(agents: Sequence[Agent], max_round: int = 50) -> GroupChat:
     """Creates a GroupChat object using the custom speaker selection."""
     if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent (like 'ProductLead').")

     return GroupChat(
         agents=list(agents),
         messages=[],
         max_round=max_round,
         speaker_selection_method=custom_speaker_selection,
         allow_repeat_speaker=False, # Standard setting
     )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    """Creates a GroupChatManager."""
    config = llm_config.get_config()
    if not config:
        raise ValueError("Invalid LLM configuration for GroupChatManager.")

    return GroupChatManager(
        groupchat=groupchat,
        llm_config=config,
         is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    )


# --- Updated Chat Initiation (Fixed - Adds initial message to history) ---
def initiate_chat_task(
    user_agent: UserProxyAgent, # This is the 'ProductLead' agent
    manager: GroupChatManager,
    initial_prompt: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Initiates the chat: resets history, adds the first message from ProductLead to the
    manager's groupchat history, and sets the first speaker.
    Returns the initial message list (for display) and the next speaker.
    """
    manager.groupchat.reset() # Ensure chat history is clear before starting
    logger.info("GroupChat history reset.")

    initial_message = {
        "role": "user",
        "content": initial_prompt.strip(),
        "name": user_agent.name
    }
    # *** Add the initial message to the manager's history ***
    manager.groupchat.messages.append(initial_message)
    logger.info(f"Initial message from {user_agent.name} added to manager history.")
    try:
        logger.debug(f"History after initial message: {json.dumps(manager.groupchat.messages)}")
    except Exception as e:
        logger.warning(f"Could not log initial message history: {e}")


    policy_guard_agent = manager.groupchat.agent_by_name("PolicyGuard")
    if not policy_guard_agent:
        logger.error("PolicyGuard agent not found in groupchat! Cannot explicitly set as first speaker.")
        logger.debug("PolicyGuard not found, selecting speaker based on initial message...")
        next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat) # Call selection
        logger.warning(f"Falling back to default speaker selection. Next: {next_speaker.name if next_speaker else 'None'}")
    else:
        next_speaker = policy_guard_agent
        logger.info(f"Explicitly selected 'PolicyGuard' as the first agent to speak.")

    return [initial_message], next_speaker


# --- Agent Step Execution (Revised Logic - WITH DEBUG LOGGING) ---
def run_agent_step(
    manager: GroupChatManager,
    speaker: Agent
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Runs a single step of the conversation for the given speaker using revised logic.
    Returns ONLY the new messages generated/added in this step and the next speaker.
    It handles cases where autogen might or might not add the message to history.
    """
    newly_added_messages = []
    next_speaker = None
    try:
        logger.info(f"--- Running step for agent: {speaker.name} ---")
        messages_context = manager.groupchat.messages
        len_before_reply = len(messages_context)
        logger.info(f"Messages history length before {speaker.name}.generate_reply: {len_before_reply}")
        if messages_context:
             try:
                 logger.debug(f" Last message before generation: {json.dumps(messages_context[-1])}")
             except Exception as e:
                 logger.warning(f"Could not log last message before generation: {e}")
        else:
            logger.debug(" Message history is empty before generation.")


        # 1. Get reply from the current speaker.
        reply = speaker.generate_reply(messages=messages_context, sender=manager)

        try:
            log_reply_content_str = json.dumps(reply, indent=2) if isinstance(reply, dict) else str(reply)
            logger.info(f"Raw reply generated by {speaker.name}: {log_reply_content_str[:500]}..." + ('...' if len(str(reply)) > 500 else ''))
        except Exception as log_e:
            logger.error(f"Error logging raw reply: {log_e}")

        # 2. Check if the message history was updated by the framework
        messages_after_reply = manager.groupchat.messages
        len_after_reply = len(messages_after_reply)
        num_new_messages = len_after_reply - len_before_reply
        logger.info(f"Messages history length after {speaker.name}.generate_reply: {len_after_reply} ({num_new_messages} new)")

        if num_new_messages > 0:
            newly_added_messages = messages_after_reply[len_before_reply:]
            logger.info(f"Captured {num_new_messages} new message(s) automatically added by framework for {speaker.name}.")
            for i, msg in enumerate(newly_added_messages):
               try:
                   logger.debug(f"  Auto-added msg {i+1}: {json.dumps(msg)}")
               except Exception: logger.debug(f"  Auto-added msg {i+1} (str): {msg}")

        elif reply is not None:
            logger.warning(f"Agent {speaker.name} generated a reply but message count didn't increase. Manually adding.")
            reply_content = None
            if isinstance(reply, dict):
                reply_content = reply.get("content")
            elif isinstance(reply, str):
                reply_content = reply
            else:
                logger.error(f"Agent {speaker.name} generated reply in unexpected format: {type(reply)}. Cannot add manually.")

            if reply_content is not None:
                role = "user" if isinstance(speaker, UserProxyAgent) else "assistant"
                manual_message = {
                    "role": role,
                    "content": reply_content,
                    "name": speaker.name
                }
                try:
                    logger.debug(f"Attempting to manually append message: {json.dumps(manual_message)}") # Added try-except
                except Exception as e:
                     logger.warning(f"Could not log manual message before append: {e}")

                manager.groupchat.messages.append(manual_message)
                logger.debug(f"Manually appended. History length now: {len(manager.groupchat.messages)}")
                try:
                    logger.debug(f" Last message in history after manual append: {json.dumps(manager.groupchat.messages[-1])}")
                except Exception as e:
                    logger.warning(f"Could not log last message after manual append: {e}")

                newly_added_messages = [manual_message]
                logger.info(f"Manually added message from {speaker.name} (Role: {role}) to history.")
            else:
                logger.warning(f"Could not extract valid content from the reply generated by {speaker.name} (reply_content is None). No message added manually.")
                newly_added_messages = []
        else:
             logger.info(f"Agent {speaker.name} generated no reply (reply is None) and no message was added automatically.")
             newly_added_messages = []

        # 3. Determine the next speaker using the updated history
        logger.debug(f"Calling manager.groupchat.select_speaker. Last speaker was {speaker.name}.")
        try:
            logger.debug(f" History state just before speaker selection: Last message: {json.dumps(manager.groupchat.messages[-1]) if manager.groupchat.messages else 'Empty'}")
        except Exception as e:
            logger.warning(f"Could not log history state before speaker selection: {e}")

        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        logger.debug(f"select_speaker returned: {next_speaker.name if next_speaker else 'None'}")
        logger.info(f"Step completed for {speaker.name}. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    except Exception as e:
        logger.error(f"Error during agent step for {speaker.name}: {e}", exc_info=True)
        try:
            product_lead_agent = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)
            if product_lead_agent:
                 next_speaker = product_lead_agent
                 logger.info(f"Error occurred. Defaulting next speaker to ProductLead: {next_speaker.name}")
            else:
                 logger.error("ProductLead agent not found! Cannot default next speaker on error.")
                 next_speaker = None
        except Exception as fallback_e:
            logger.error(f"Failed to fallback to ProductLead agent after error: {fallback_e}")
            next_speaker = None

    return newly_added_messages, next_speaker


# --- User Message Sending (Revised to align with run_agent_step/manual add - No changes needed here) ---
def send_user_message(
    manager: GroupChatManager,
    user_agent: UserProxyAgent, # ProductLead agent
    user_message: str
    ) -> Tuple[List[Dict], Optional[Agent]]:
    """
    Handles sending a message from the user (ProductLead) into the chat.
    Manually constructs and adds the message to the groupchat history.
    Returns the message added and the next speaker determined by the selection method.
    """
    if not user_message or not user_message.strip():
        logger.warning("Empty user message received. Attempting to re-select speaker without adding message.")
        last_actual_speaker = None
        if manager.groupchat.messages:
             try:
                 last_msg_name = manager.groupchat.messages[-1].get('name')
                 if last_msg_name:
                      last_actual_speaker = manager.groupchat.agent_by_name(last_msg_name)
             except Exception as e:
                  logger.warning(f"Could not determine last speaker from history: {e}")
        if not last_actual_speaker:
             last_actual_speaker = user_agent

        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        logger.info(f"Empty message ignored. Recalculated next speaker: {next_speaker.name if next_speaker else 'None'}")
        return [], next_speaker

    message_dict = {
        "role": "user",
        "content": user_message.strip(),
        "name": user_agent.name
    }
    try:
        logger.debug(f"Attempting to manually append user message: {json.dumps(message_dict)}") # Added try-except
    except Exception as e:
         logger.warning(f"Could not log user message before append: {e}")

    manager.groupchat.messages.append(message_dict)
    logger.debug(f"Manually appended user message. History length now: {len(manager.groupchat.messages)}")
    try:
        logger.debug(f" Last message in history after user append: {json.dumps(manager.groupchat.messages[-1])}")
    except Exception as e:
        logger.warning(f"Could not log last message after user append: {e}")

    logger.info(f"User message from {user_agent.name} manually added to history.")

    logger.debug(f"Calling manager.groupchat.select_speaker after user message. Last speaker was {user_agent.name}.")
    try:
        logger.debug(f" History state just before speaker selection: Last message: {json.dumps(manager.groupchat.messages[-1]) if manager.groupchat.messages else 'Empty'}")
    except Exception as e:
        logger.warning(f"Could not log history state before speaker selection: {e}")

    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    logger.debug(f"select_speaker returned: {next_speaker.name if next_speaker else 'None'}")
    logger.info(f"User message sent. Selected next speaker: {next_speaker.name if next_speaker else 'None'}")

    return [message_dict], next_speaker
