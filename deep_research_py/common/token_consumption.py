from dataclasses import dataclass

@dataclass
class TokenUsageEvent:
    event: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int

    def __repr__(self):
        return (
            f"TokenUsageEvent(event={self.event}, "
            f"input_tokens={self.input_tokens}, "
            f"output_tokens={self.output_tokens}, "
            f"reasoning_tokens={self.reasoning_tokens})"
        )

class TokenCounter:
    def __init__(self):
        self.token_usage = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_reasoning_tokens = 0

    def add_event(self, event: TokenUsageEvent):
        self.token_usage.append(event)
        self.total_input_tokens += event.input_tokens
        self.total_output_tokens += event.output_tokens
        self.total_reasoning_tokens += event.reasoning_tokens

    def __repr__(self):
        return (
            f"TokenCounter(total_input_tokens={self.total_input_tokens}, "
            f"total_output_tokens={self.total_output_tokens}, "
            f"total_reasoning_tokens={self.total_reasoning_tokens})\n"
            "Events: \n" + "\n".join([str(event) for event in self.token_usage])
        )

counter = TokenCounter()

def count_token_consume(event: str, input_tokens: int, output_tokens: int, reasoning_tokens: int):
    from common.token_consumption import TokenUsageEvent, counter
    token_event = TokenUsageEvent(event, input_tokens, output_tokens, reasoning_tokens)
    counter.add_event(token_event)

def parse_openai_token_consume(event: str, response):
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    # For simplicity, set reasoning tokens to 0.
    reasoning_tokens = 0
    count_token_consume(
        event=event, input_tokens=input_tokens, output_tokens=output_tokens, reasoning_tokens=reasoning_tokens
    )
