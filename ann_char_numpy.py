import numpy as np
import sys

text = sys.argv[1] if len(sys.argv) > 1 else "hello world"

# One training example for every adjacent pair:
# "h" -> "e", "e" -> "l", ...
chars = sorted(set(text))
n = len(chars)
char_to_i = {c: i for i, c in enumerate(chars)}
i_to_char = {i: c for i, c in enumerate(chars)}

# One-hot character input and target
x = np.eye(n)[[char_to_i[c] for c in text[:-1]]]
y = np.eye(n)[[char_to_i[c] for c in text[1:]]]

# Same basic network as ann_numpy.py:
# character -> hidden layer -> next character
hidden_size = max(4, n)
rng = np.random.default_rng(1)

w1 = rng.normal(0, 0.5, (hidden_size, n))
w2 = rng.normal(0, 0.5, (n, hidden_size))
bias1 = np.zeros(hidden_size)
bias2 = np.zeros(n)

lr = 0.5
iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 10000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


for iteration in range(iterations):

    # Forward pass
    h = sigmoid(x.dot(w1.T) + bias1)
    y_pred = sigmoid(h.dot(w2.T) + bias2)

    # Backpropagation, same idea as ann_numpy.py
    delta2 = (y_pred - y) * y_pred * (1 - y_pred)
    w2 = w2 - lr * delta2.T.dot(h)
    bias2 = bias2 - lr * delta2.sum(axis=0)

    delta1 = delta2.dot(w2) * h * (1 - h)
    w1 = w1 - lr * delta1.T.dot(x)
    bias1 = bias1 - lr * delta1.sum(axis=0)

    error = 0.5 * np.square(y_pred - y).sum()

    if (iteration + 1) % 1000 == 0 or iteration == 0:
        print(f"Iteration {iteration + 1}: Error = {error:.10f}")


def predict(c):
    x = np.eye(n)[char_to_i[c]]
    h = sigmoid(x.dot(w1.T) + bias1)
    p = sigmoid(h.dot(w2.T) + bias2)
    return i_to_char[np.argmax(p)]


# Show what the network learned for every character.
print("\nNext-character predictions:")
for c in chars:
    print(repr(c), "->", repr(predict(c)))


# Generate text by repeatedly predicting the next character.
result = text[0]
c = text[0]

for _ in range(max(0, len(text) - 1)):
    c = predict(c)
    result += c

print("\nInput:     ", repr(text))
print("Generated: ", repr(result))
