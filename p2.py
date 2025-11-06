import numpy as np
class HeteroAssociativeMemory:
    def __init__(self, input_size, output_size):
        self.weights = np.zeros((output_size, input_size))

    def train(self, input_patterns, output_patterns):
        for x, y in zip(input_patterns, output_patterns):
            x = np.array(x)
            y = np.array(y)
            self.weights += np.outer(y, x)
    def recall(self, input_pattern):
        input_pattern = np.array(input_pattern)
        output = self.weights @ input_pattern
        return np.sign(output)
if __name__ == "__main__":
    inputs = [
        [1, -1, 1],
        [-1, -1, 1]
    ]
    outputs = [
        [-1, 1],
        [1, 1]
    ]

    model = HeteroAssociativeMemory(input_size=3, output_size=2)
    model.train(inputs, outputs)
    test_input = [1, -1, 1]
    recalled = model.recall(test_input)
    print("Input:    ", test_input)
    print("Recalled: ", recalled.tolist())
