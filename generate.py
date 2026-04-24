import torch
import torch.nn as nn
from torchtyping import TensorType
import torch.nn.functional as F

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.
        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = ""
        for i in range(new_chars):
            # YOUR CODE (arbitrary number of lines)
            # The line where you call torch.multinomial(). Pass in the generator as well.
            context = context[:, -context_length:]
            logits = model(context)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            generator.set_state(initial_state)
            next_token = torch.multinomial(probs, 1, generator=generator)
            torch.cat([context, next_token], dim=1)
            # MORE OF YOUR CODE (arbitrary number of lines)
            result += int_to_char[int(next_token[0, 0].item())]
        return result
            

        # Once your code passes the test, check out the Colab link to see your code generate new Drake lyrics!
