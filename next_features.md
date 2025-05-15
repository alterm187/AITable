# Next Features Roadmap

1. Introduce model config file with models and costs. Make the model choice possible per persona. Make sure Google, Anthropic and OpenAI models are possible to be chosen [In progress]
2. Introduce cost calculations for the current conversation based on tokens consumed and the analysis of how many times different personas were engaged. Transform the cost into PLN based on current rate
3. Plan a feature of turning on 'thinking' mode of a persona-model in flight, druing conversation, after the user says something like "think it through" or "przeanalizuj". The example list of such phrases need to be prepared.
4. Plan a feature of attaching google docs text file to a conversation to be visible in a chat history
5. Plan a feature of summarizing older messages in the current chat context after crossing some token number threshold or another border
6. Plan a feature of keeping memory track between chat sessions in a separate vectorDB