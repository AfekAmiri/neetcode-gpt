from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        tokenized: List[List[str]] = []

        for num in numbers:
            s = str(num)
            tokens: List[str] = []
            j = 0

            while j < len(s):
                best = None

                # Find the longest token starting at position j
                for i in range(j + 1, len(s) + 1):
                    piece = s[j:i]
                    if piece in vocab:
                        best = piece

                # If nothing matched, fall back to one character to avoid infinite loop
                if best is None:
                    best = s[j:j + 1]

                tokens.append(best)
                j += len(best)

            tokenized.append(tokens)
        return tokenized

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        num_tokens = 0
        i = 0 
        while i < len(text):
            best = None

            # Find longest vocab token starting at i
            for j in range(i + 1, len(text) + 1):
                piece = text[i:j]
                if piece in vocab:
                    best = piece

            # Fallback to single char if no token matched
            if best is None:
                best = text[i:i + 1]

            num_tokens += 1
            i += len(best)
        return num_tokens
            


    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words = text.split()
        token_count = self.count_tokens(text, vocab)
        fertility = token_count / len(words)
        return round(fertility, 4)
