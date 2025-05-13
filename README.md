# Project: Prototype Application with PolicyGuard and Challenger

This project is a prototype application of a groupchat based on AutoGen framework. The idea is to have PolicyGuard agent with set of policies attached to be followed and instructed how it should analyze if the given product description (provided with the task) is matching the policy or how to make it matching.

The second agent, Challenger, is instructed how to challenge the analysis and recommendations provided by PolicyGuard. User agent is user proxy allowing human to participate in the discussion.


## LLM Integration and Setup

This application leverages Large Language Models (LLMs) for agent interaction and task completion, utilizing the AutoGen framework. It supports integration with both OpenAI and Google Cloud Platform (GCP) Vertex AI.

**Supported LLMs:**

*   **OpenAI:** For using OpenAI models, you need to have an OpenAI API key.
*   **GCP/Vertex AI:** For using Vertex AI models, you need to have a GCP project ID and location.

**Initialization and Configuration:**

1.  **AutoGen:** The project is built using the AutoGen framework, which simplifies the orchestration of multi-agent workflows.
2.  **OpenAI API Key:** If you plan to use OpenAI models, set your API key in the environment variable `OPENAI_API_KEY`.

    
## Overview

This application builds upon a previous project that included additional agent types like `Inspector` and `ContentManager`. However, the core concept remains consistent:

1.  A `groupchat` is initialized with a specific task describing a product or product concept that is supposed to be a subject of agents' discussion.
2.  Designated agents are assigned to handle the task within the groupchat.
3. System messages for `PolicyGuard` and `Challenger` need to be rewritten and adapted for this application.


