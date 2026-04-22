from typing import List
from collections import Counter


class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed

        vocab = list(corpus)
        merges: List[List[str]] = []

        for _ in range (num_merges):

            pairs = [(vocab[i],vocab[i+1]) for i in range(len(vocab)-1)]
            freq = Counter(pairs)
            best_pair = min(freq.keys(), key=lambda p: (-freq[p], p))
            merges.append([best_pair[0], best_pair[1]])

            merged = []
            i = 0
            while i < len(vocab):
                if i < len(vocab) - 1 and (vocab[i], vocab[i + 1]) == best_pair:
                    merged.append(vocab[i] + vocab[i + 1])
                    i += 2
                else:
                    merged.append(vocab[i])
                    i += 1

            vocab = merged

        return merges

            


