import logging
import re # Added import
from typing import Dict, List, Optional, Any # Added Any
from google.oauth2 import service_account
from google.cloud import aiplatform

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for model providers (USING UPPERCASE)
VERTEX_AI = "VERTEX_AI"
AZURE = "AZURE"
ANTHROPIC = "ANTHROPIC"

# Path to the model configurations file
MODELS_CONFIG_PATH = "AITable/models_config.md"

# Cache for model configurations
_model_configurations_cache: Optional[Dict[str, Dict[str, Any]]] = None

def load_model_configurations(file_path: str = MODELS_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    """Loads model configurations from a Markdown file."""
    global _model_configurations_cache
    if _model_configurations_cache is not None:
        return _model_configurations_cache

    configs: Dict[str, Dict[str, Any]] = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split models by "---"
        model_sections = content.strip().split('
---
')
        
        for section in model_sections:
            if not section.strip():
                continue

            model_name_match = re.search(r"^# Model: (\S+)", section, re.MULTILINE)
            if not model_name_match:
                logger.warning(f"Could not find model name in section: {section[:50]}...")
                continue
            model_name = model_name_match.group(1)
            
            details: Dict[str, Any] = {"model_name": model_name}
            
            display_name_match = re.search(r"Display Name: (.*)", section)
            if display_name_match:
                details["display_name"] = display_name_match.group(1).strip()
            
            input_price_match = re.search(r"Input Token Price Per 1k: (\d+\.?\d*)", section)
            if input_price_match:
                details["input_token_price_per_1k"] = float(input_price_match.group(1))
                
            output_price_match = re.search(r"Output Token Price Per 1k: (\d+\.?\d*)", section)
            if output_price_match:
                details["output_token_price_per_1k"] = float(output_price_match.group(1))
                
            context_window_match = re.search(r"Context Window: (\d+)", section)
            if context_window_match:
                details["context_window"] = int(context_window_match.group(1))
            
            if len(details) > 1: # Ensure more than just model_name was added
                configs[model_name] = details
            else:
                logger.warning(f"Could not parse details for model: {model_name}")

    except FileNotFoundError:
        logger.error(f"Model configuration file not found: {file_path}")
        return {} # Return empty if file not found, or raise error
    except Exception as e:
        logger.error(f"Error parsing model configuration file {file_path}: {e}")
        return {} # Return empty on other errors

    _model_configurations_cache = configs
    return configs

def get_available_model_display_names() -> List[str]:
    """Returns a list of display names for all configured models."""
    configs = load_model_configurations()
    return [details["display_name"] for details in configs.values() if "display_name" in details]

def get_model_config_details(model_name: str) -> Optional[Dict[str, Any]]:
    """Returns the full configuration for a given model_name."""
    configs = load_model_configurations()
    return configs.get(model_name)

def get_model_name_from_display_name(display_name: str) -> Optional[str]:
    """Returns the model_name (internal ID) from a display_name."""
    configs = load_model_configurations()
    for model_id, details in configs.items():
        if details.get("display_name") == display_name:
            return model_id
    return None


class LLMConfiguration:
    """Generic LLM configuration class."""

    def __init__(self, provider: str, model: str, **kwargs):
        self.provider = provider
        self.model = model # This 'model' is the specific model identifier like 'gpt-3.5-turbo'
        self.config = kwargs
        # Potentially enrich self.config with details from models_config.md here
        # For example, to set max_tokens based on the selected model's context_window
        model_details = get_model_config_details(self.model)
        if model_details:
            # Override max_tokens from general kwargs if specific model has context_window
            # And if 'max_tokens' was not already set to something more specific in kwargs
            if "context_window" in model_details and "max_tokens" not in self.config:
                 # Autogen's max_tokens is for completion, so it shouldn't be the full context window.
                 # Users might want to specify this separately. For now, let's use a portion or a default.
                 # Or, this 'max_tokens' in LLMConfiguration could be *output* max_tokens.
                 # Let's assume for now it's about output tokens and keep it as passed or default.
                 pass # Decide on how to use model_details["context_window"] here if needed.
            
            # We might want to store the full model_details for later use (e.g. pricing)
            self.detailed_model_config = model_details
        else:
            logger.warning(f"Could not load details for model {self.model} from models_config.md")
            self.detailed_model_config = None


    def get_config(self) -> Dict:
        """Returns the LLM configuration dictionary."""

        if self.provider == VERTEX_AI:
            return self._get_vertex_ai_config()
        elif self.provider == AZURE:
            return self._get_azure_config()
        elif self.provider == ANTHROPIC:  # Add Anthropic case
            return self._get_anthropic_config()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_vertex_ai_config(self) -> Dict:
        """Constructs and returns the Vertex AI configuration."""

        project_id = self.config.get("project_id")
        location = self.config.get("location")
        credentials_dict = self.config.get("vertex_credentials")

        if not all([project_id, location, credentials_dict]):
            raise ValueError("Missing required parameters for Vertex AI: project_id, location, vertex_credentials")

        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        aiplatform.init(
            project=project_id,
            location=location,
            credentials=credentials
        )
        
        # Determine max_tokens: use specific from model_config if available, else default
        default_max_output_tokens = 4096 # A general default
        max_output_tokens = self.config.get("max_tokens") # from __init__ kwargs
        if self.detailed_model_config and "max_output_tokens" in self.detailed_model_config: # if specified in models_config.md
             max_output_tokens = self.detailed_model_config["max_output_tokens"]
        elif max_output_tokens is None: # if not in kwargs
             max_output_tokens = default_max_output_tokens


        config = {
            "config_list": [
                {
                    "model": self.model,
                    "api_type": "google",
                    "location": location,
                    "project_id": project_id,
                }
            ],
            "cache_seed": self.config.get("cache_seed", 42),
            "temperature": self.config.get("temperature", 0),
            "max_tokens": max_output_tokens, # Use the determined max_output_tokens
        }
        return config

    def _get_azure_config(self) -> Dict:
        """Constructs and returns the Azure configuration."""

        required_params = ["api_key", "base_url", "api_version"]
        if not all(param in self.config for param in required_params):
            raise ValueError(f"Missing required parameters for Azure: {required_params}")

        default_max_output_tokens = 4096
        max_output_tokens = self.config.get("max_tokens")
        if self.detailed_model_config and "max_output_tokens" in self.detailed_model_config:
             max_output_tokens = self.detailed_model_config["max_output_tokens"]
        elif max_output_tokens is None:
             max_output_tokens = default_max_output_tokens

        config = {
            "config_list": [
                {
                    "model": self.model, # This should be the deployment name for Azure
                    "api_key": self.config["api_key"],
                    "base_url": self.config["base_url"],
                    "api_type": "azure",
                    "api_version": self.config["api_version"],
                    "max_tokens": max_output_tokens, 
                }
            ],
            "temperature": self.config.get("temperature", 0),
        }
        return config

    def _get_anthropic_config(self) -> Dict:
        """Constructs and returns the Anthropic configuration."""

        required_params = ["api_key"] # base_url is often not needed if using official SDK client
        if not all(param in self.config for param in required_params):
            raise ValueError(f"Missing required parameters for Anthropic: {required_params}")
        
        default_max_output_tokens = 4096
        max_output_tokens = self.config.get("max_tokens")
        if self.detailed_model_config and "max_output_tokens" in self.detailed_model_config:
             max_output_tokens = self.detailed_model_config["max_output_tokens"]
        elif max_output_tokens is None:
             max_output_tokens = default_max_output_tokens

        config = {
            "config_list": [
                {
                    "model": self.model,
                    "api_key": self.config["api_key"],
                    "api_type": "anthropic",
                    "max_tokens": max_output_tokens, # Anthropic uses max_tokens_to_sample
                }
            ],
            "request_timeout": self.config.get("request_timeout", 120),
            "temperature": self.config.get("temperature", 0),
        }
        # Anthropic uses 'max_tokens_to_sample' in some contexts, autogen might map 'max_tokens' to it.
        # If direct control is needed, this might need adjustment:
        # config["config_list"][0]["max_tokens_to_sample"] = max_output_tokens
        # And remove "max_tokens" if it causes issues.
        # For now, assume autogen handles the mapping for "max_tokens".
        return config
