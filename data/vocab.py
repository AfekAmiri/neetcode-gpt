from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)

        vocab = sorted(set(text))
        stoi = {char: i for i, char in enumerate(vocab)}
        itos = {i: char for i, char in enumerate(vocab)}

        return stoi, itos

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        encoded_text = [stoi[c] for c in text]
        return encoded_text

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        decoded_ids = "".join(itos[id] for id in ids)
        return decoded_ids
