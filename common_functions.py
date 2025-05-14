import logging
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChatManager, Agent, GroupChat
from typing import List, Optional, Sequence, Tuple, Dict, Union # Union is important here
import json
import re
import functools
import os

from LLMConfiguration import LLMConfiguration, logger

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.DEBUG)

def load_personas(persona_dir: str, persona_filenames: List[str]) -> str:
    combined_system_message = []
    if not persona_filenames: return ""
    for filename in persona_filenames:
        try:
            safe_filename = os.path.basename(filename)
            if not safe_filename.endswith(".md"): continue
            filepath = os.path.join(persona_dir, safe_filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content: combined_system_message.append(content)
        except Exception as e:
            module_logger.error(f"Error reading persona file {filename}: {e}")
    return "\n---\n".join(combined_system_message)

def create_agent(
    name: str, llm_config: LLMConfiguration,
    system_message_content: Optional[str] = None,
    agent_type="assistant"
) -> autogen.Agent:
    system_message_for_agent = (system_message_content or "").strip() or "You are a helpful assistant."
    config = llm_config.get_config()
    if not config: raise ValueError(f"Invalid LLM config for {name}")
    is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
    if agent_type == "user_proxy":
        return UserProxyAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", code_execution_config=False, llm_config=config, default_auto_reply="", is_termination_msg=is_termination_msg)
    else:
        return AssistantAgent(name=name, system_message=system_message_for_agent, human_input_mode="NEVER", llm_config=config, is_termination_msg=is_termination_msg)

def read_system_message(filename: str) -> Tuple[Optional[str], str]:
    default_system_message = "You are a helpful assistant."
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            full_content = file.read()
            lines = full_content.splitlines()
            display_name = None
            system_message_content = full_content.strip()
            if lines:
                match = re.match(r"^DisplayName:\s*(.+)$", lines[0].strip(), re.IGNORECASE)
                if match:
                    display_name = match.group(1).strip()
                    system_message_content = "\n".join(lines[1:]).strip()
            if not system_message_content: 
                system_message_content = default_system_message
                if display_name: module_logger.warning(f"Content empty for {filename} with DisplayName, using default content.")
                else: module_logger.warning(f"File {filename} empty or only DisplayName. Using default content.")
            return display_name, system_message_content
    except FileNotFoundError:
        return None, default_system_message
    except Exception as e:
        module_logger.error(f"Error reading {filename}: {e}")
        return None, default_system_message

def custom_speaker_selection(
    last_speaker: Agent, 
    groupchat: GroupChat, 
    # agent_display_config can now have str or List[str] as values
    agent_display_config: Dict[str, Union[str, List[str]]]
) -> Agent:
    module_logger.debug(f"CustomSpeakerSelection: Last speaker: {last_speaker.name}")
    user_agent = next((agent for agent in groupchat.agents if isinstance(agent, UserProxyAgent)), None)
    if not user_agent: raise ValueError("UserProxyAgent not found in groupchat agents.")

    if not groupchat.messages:
        module_logger.debug("No messages, User speaks first.")
        return user_agent

    last_message_content = str(groupchat.messages[-1].get('content', '')).strip()
    if last_message_content.rstrip().endswith("TERMINATE"): 
        module_logger.debug("TERMINATE detected. User's turn.")
        return user_agent

    lower_message_content = last_message_content.lower()
    best_mention_index = -1
    longest_mention_len = -1
    agent_to_select = None

    for agent in groupchat.agents:
        if isinstance(agent, UserProxyAgent): continue # User isn't selected by mention here

        agent_code_name = agent.name
        display_names_or_aliases = agent_display_config.get(agent_code_name)
        
        if not display_names_or_aliases: 
            module_logger.warning(f"No display name/aliases for '{agent_code_name}' in config.")
            continue

        # Ensure it's a list for consistent iteration
        if isinstance(display_names_or_aliases, str):
            search_patterns = [display_names_or_aliases.lower()]
        else: # It's a List[str]
            search_patterns = [name.lower() for name in display_names_or_aliases if name.strip()]

        for pattern in search_patterns:
            if not pattern: continue
            
            # Find the last occurrence of this specific pattern
            current_pattern_last_idx = -1
            start_search_from = 0
            while True:
                found_idx = lower_message_content.find(pattern, start_search_from)
                if found_idx == -1: break
                current_pattern_last_idx = found_idx
                start_search_from = found_idx + len(pattern)

            if current_pattern_last_idx > best_mention_index:
                best_mention_index = current_pattern_last_idx
                longest_mention_len = len(pattern)
                agent_to_select = agent
                module_logger.debug(f"New best mention: '{pattern}' for '{agent_code_name}' at {best_mention_index}.")
            elif current_pattern_last_idx == best_mention_index and current_pattern_last_idx != -1:
                if len(pattern) > longest_mention_len:
                    longest_mention_len = len(pattern)
                    agent_to_select = agent
                    module_logger.debug(f"Same index, longer mention: '{pattern}' for '{agent_code_name}'. Switched.")
    
    if agent_to_select:
        if agent_to_select == last_speaker:
            module_logger.debug(f"Mentioned agent '{agent_to_select.name}' is last speaker. Defaulting to User.")
            return user_agent
        else:
            module_logger.debug(f"Speaker by mention: '{agent_to_select.name}'.")
            return agent_to_select
    else:
        module_logger.debug("No specific agent mentioned. Defaulting to User.")
        return user_agent

def create_groupchat(
    agents: Sequence[Agent], 
    # This param now expects Dict[str, Union[str, List[str]]]
    agent_display_config: Dict[str, Union[str, List[str]]], 
    max_round: int = 50
) -> GroupChat:
    if not any(isinstance(agent, UserProxyAgent) for agent in agents):
         raise ValueError("GroupChat requires at least one UserProxyAgent.")
    
    # Pass the agent_display_config (which can contain aliases) to custom_speaker_selection
    speaker_selection_func = functools.partial(custom_speaker_selection, agent_display_config=agent_display_config)

    return GroupChat(
        agents=list(agents),
        messages=[],
        max_round=max_round,
        speaker_selection_method=speaker_selection_func,
        allow_repeat_speaker=False 
    )

def create_groupchat_manager(groupchat: GroupChat, llm_config: LLMConfiguration) -> GroupChatManager:
    config = llm_config.get_config()
    if not config: raise ValueError("Invalid LLM configuration for GroupChatManager.")
    return GroupChatManager(groupchat=groupchat, llm_config=config, is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"))

def initiate_chat_task(
    user_agent: UserProxyAgent, manager: GroupChatManager, initial_prompt: str,
    system_content_for_group: Optional[str] = None
) -> Tuple[List[Dict], Optional[Agent]]:
    manager.groupchat.reset()
    final_prompt = initial_prompt.strip()
    if system_content_for_group and system_content_for_group.strip():
        final_prompt = f"General Context:\n{system_content_for_group.strip()}\n---\nUser's Task:\n{final_prompt}"
    
    initial_message = {"role": "user", "content": final_prompt, "name": user_agent.name}
    manager.groupchat.messages.append(initial_message)
    module_logger.debug(f"Initial message from {user_agent.name} added.")

    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.debug(f"Initiate chat: First AI speaker: {next_speaker.name if next_speaker else 'None'}")
    return [initial_message], next_speaker

def run_agent_step(manager: GroupChatManager, speaker: Agent) -> Tuple[List[Dict], Optional[Agent]]:
    newly_added_messages = []
    next_speaker = None
    user_agent = next((agent for agent in manager.groupchat.agents if isinstance(agent, UserProxyAgent)), None)

    try:
        module_logger.debug(f"Running step for agent: {speaker.name}")
        messages_context = manager.groupchat.messages
        len_before_reply = len(messages_context)
        reply = speaker.generate_reply(messages=messages_context, sender=manager)
        
        if len(manager.groupchat.messages) > len_before_reply:
            newly_added_messages = manager.groupchat.messages[len_before_reply:]
        elif reply is not None:
            reply_content = reply.get("content") if isinstance(reply, dict) else str(reply)
            if reply_content:
                role = "user" if isinstance(speaker, UserProxyAgent) else "assistant"
                manual_message = {"role": role, "content": reply_content, "name": speaker.name}
                manager.groupchat.messages.append(manual_message)
                newly_added_messages = [manual_message]
            else:
                module_logger.warning(f"Empty reply content from {speaker.name}.")
        else:
             module_logger.info(f"Agent {speaker.name} generated no reply.")

        next_speaker = manager.groupchat.select_speaker(speaker, manager.groupchat)
        module_logger.debug(f"Step for {speaker.name} done. Next: {next_speaker.name if next_speaker else 'None'}")
    except Exception as e:
        module_logger.error(f"Error in agent step for {speaker.name}: {e}", exc_info=True)
        if user_agent: next_speaker = user_agent
    return newly_added_messages, next_speaker

def send_user_message(
    manager: GroupChatManager, user_agent: UserProxyAgent, user_message: str
) -> Tuple[List[Dict], Optional[Agent]]:
    user_message_stripped = user_message.strip()
    if not user_message_stripped:
        module_logger.debug("User sent empty message. Selecting next speaker.")
        last_actual_speaker = user_agent
        if manager.groupchat.messages:
            try: 
                last_msg_name = manager.groupchat.messages[-1].get('name')
                if last_msg_name: last_actual_speaker = manager.groupchat.agent_by_name(last_msg_name) or user_agent
            except: pass
        next_speaker = manager.groupchat.select_speaker(last_actual_speaker, manager.groupchat)
        return [], next_speaker

    message_dict = {"role": "user", "content": user_message_stripped, "name": user_agent.name}
    manager.groupchat.messages.append(message_dict)
    next_speaker = manager.groupchat.select_speaker(user_agent, manager.groupchat)
    module_logger.debug(f"User message sent. Next speaker: {next_speaker.name if next_speaker else 'None'}")
    return [message_dict], next_speaker
