#Implement Auto-Associative Memory Network for pattern storage and retrieval.
import numpy as np
class AutoAssociativeMemory:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
    def train(self, patterns):
        for p in patterns:
            p = np.array(p)
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)  
    def recall(self, pattern, steps=5):
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
            pattern[pattern == 0] = 1
        return pattern
if __name__ == "__main__":
    patterns = [
        [1, -1, 1, -1],
        [-1, -1, 1, 1]
    ]
    model = AutoAssociativeMemory(size=4)
    model.train(patterns)
    test_pattern = [1, -1, -1, -1]
    recalled = model.recall(test_pattern)
    print("Input:    ", test_pattern)
    print("Recalled: ", recalled.tolist())

