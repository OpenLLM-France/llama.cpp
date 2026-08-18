"""
Test conversations used to compare the transformers chat template (jinja)
against the Ollama Modelfile template.

Each test case is a dict with:
    name      : unique short identifier (used in filenames and reports)
    messages  : OpenAI-style list of message dicts
    tools     : list of OpenAI-style tool definitions, or None
"""

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris'.",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit.",
                },
            },
            "required": ["location"],
        },
    },
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a math expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A math expression, e.g. '2 + 2 * 3'.",
                },
            },
            "required": ["expression"],
        },
    },
}


TEST_CASES = [
    # --- No tools ---
    {
        "name": "01_user_only",
        "messages": [
            {"role": "user", "content": "Bonjour, comment vas-tu ?"},
        ],
        "tools": None,
    },
    {
        "name": "02_system_user",
        "messages": [
            {"role": "system", "content": "Tu es un assistant qui répond en français."},
            {"role": "user", "content": "Quelle est la capitale de la France ?"},
        ],
        "tools": None,
    },
    {
        "name": "03_multi_turn",
        "messages": [
            {"role": "user", "content": "Hi!"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "What's 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "Thanks!"},
        ],
        "tools": None,
    },
    # --- Tools, no tool call yet ---
    {
        "name": "04_tools_available_no_call",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        "tools": [WEATHER_TOOL],
        # Behavioural expectation: the model should emit a tool_call rather than text.
        "expected_behavior": {
            "tool_call": {
                "name": "get_weather",
                "required_args": ["location"],
                "args_must_contain": {"location": "paris"},  # case-insensitive substring
            },
        },
    },
    {
        "name": "05_tools_with_system",
        "messages": [
            {"role": "system", "content": "You are a weather assistant."},
            {"role": "user", "content": "Weather in Paris please."},
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
        "expected_behavior": {
            "tool_call": {
                "name": "get_weather",
                "required_args": ["location"],
                "args_must_contain": {"location": "paris"},
            },
        },
    },
    # --- Tool call + tool response ---
    {
        "name": "06_single_tool_call_and_response",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 18, "unit": "celsius"}'},
            {"role": "assistant", "content": "It's 18°C in Paris."},
        ],
        "tools": [WEATHER_TOOL],
    },
    # --- Multiple tool calls in one assistant turn ---
    {
        "name": "07_multiple_tool_calls",
        "messages": [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ],
            },
        ],
        "tools": [WEATHER_TOOL],
    },
    # --- Consecutive tool responses (must batch into one user turn in jinja) ---
    {
        "name": "08_consecutive_tool_responses",
        "messages": [
            {"role": "user", "content": "Weather in Paris and London?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "London"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"location": "Paris", "temperature": 18}'},
            {"role": "tool", "content": '{"location": "London", "temperature": 15}'},
        ],
        "tools": [WEATHER_TOOL],
    },
    # --- Assistant with BOTH content AND tool_calls ---
    {
        "name": "09_assistant_content_and_tool_call",
        "messages": [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": "Let me check that for you.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Paris"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 18, "unit": "celsius"}'},
        ],
        "tools": [WEATHER_TOOL],
    },
    # --- Long-ish multi-turn with tools sprinkled in ---
    {
        "name": "10_full_tool_dialogue",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "Compute 12*34 then tell me the weather in Lyon."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "12*34"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"result": 408}'},
            {
                "role": "assistant",
                "content": "12*34 = 408. Now checking the weather.",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Lyon"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": '{"temperature": 21, "unit": "celsius"}'},
            {"role": "assistant", "content": "12*34 = 408 and it's 21°C in Lyon."},
        ],
        "tools": [WEATHER_TOOL, CALCULATOR_TOOL],
    },
    # --- Multi-turn (user/assistant/user/assistant/user), tokenization-focused ---
    # Mixes French accents, an apostrophe word ("l'Allemagne"), a code fence
    # and a Markdown inline code span — designed to surface BPE / special-token
    # boundary divergences between the HF tokenizer and llama.cpp's SPM.
    {
        "name": "11_multi_turn_tokenization",
        "messages": [
            {"role": "user", "content": "Quelle heure est-il à Paris ?"},
            {"role": "assistant", "content": "Il est 14h30 (heure de Paris)."},
            {"role": "user", "content": "Montre-moi `int x = 42;` dans un bloc Markdown."},
            {"role": "assistant", "content": "```cpp\nint x = 42;\n```"},
            {"role": "user", "content": "Merci !"},
        ],
        "tools": None,
    },
    # --- Multi-turn (user/assistant/user/assistant/user), logit-focused ---
    # Short factual conversation ending on a question whose answer is a single
    # well-known proper noun ("Madrid"), so the next-token top-K distribution
    # should be sharply peaked and easy to compare across runtimes.
    {
        "name": "12_multi_turn_logit",
        "messages": [
            {"role": "user", "content": "Quelle est la capitale de la France ?"},
            {"role": "assistant", "content": "La capitale de la France est Paris."},
            {"role": "user", "content": "Et celle de l'Allemagne ?"},
            {"role": "assistant", "content": "Berlin."},
            {"role": "user", "content": "Et celle de l'Espagne ?"},
        ],
        "tools": None,
    },
]
