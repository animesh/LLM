# LLM

## Phase 0

ANN https://github.com/animesh/ANN to LLM where original network is essentially:

`2 numbers → 2 hidden neurons → 2 output neurons`

The character version becomes:

`character → hidden neurons → next character`

For example, with:

```text
hello world
```

the training pairs are:

```text
h → e
e → l
l → l
l → o
o → ' '
' ' → w
w → o
o → r
r → l
l → d
```

Each character is represented as a one-hot vector. So with four characters, for example:

```text
a = [1,0,0,0]
b = [0,1,0,0]
c = [0,0,1,0]
d = [0,0,0,1]
```

The output is another vector of the same size, and `argmax()` selects the predicted next character.

The complete implementation is here: [Download ann_char_numpy.py](ann_char_numpy.py)

Example run:

```bash
python ann_char_numpy.py "hello world" 10000
Iteration 1: Error = 9.5553823938
Iteration 1000: Error = 1.5082447006
Iteration 2000: Error = 1.5036012548
Iteration 3000: Error = 1.5022734563
Iteration 4000: Error = 1.5016522991
Iteration 5000: Error = 1.5012941363
Iteration 6000: Error = 1.5010617911
Iteration 7000: Error = 1.5008991621
Iteration 8000: Error = 1.5007791101
Iteration 9000: Error = 1.5006869306
Iteration 10000: Error = 1.5006139741

Next-character predictions:
' ' -> 'w'
'd' -> 'l'
'e' -> 'l'
'h' -> 'e'
'l' -> 'l'
'o' -> ' '
'r' -> 'l'
'w' -> 'o'

Input:      'hello world'
Generated:  'helllllllll'
```

This deliberately very close to 

```python
h = sigmoid(x.dot(w1.T) + bias1)
y_pred = sigmoid(h.dot(w2.T) + bias2)

delta2 = (y_pred - y) * y_pred * (1 - y_pred)

w2 = w2 - lr * delta2.T.dot(h)

delta1 = delta2.dot(w2) * h * (1 - h)

w1 = w1 - lr * delta1.T.dot(x)
```

## Phase 1

One conceptual limitation worth making explicit: **this is not yet an LLM**. It is a minimal character-level neural language model with no memory. It learns:

```text
P(next_character | current_character)
```

not:

```text
P(next_character | all_previous_characters)
```

So the next step while preserving minimal-from-scratch approach:

```text
"hello" → " "
       ↓
   character sequence
       ↓
   hidden layer
       ↓
 next character
```

and then introduce a small context window, e.g.

```text
hel → l
ell → o
llo → " "
lo  → ...
```

That will probably gets us much closer to the fundamental idea behind an autoregressive language model, while still allowing us to derive every operation and backpropagation step from the code [ANN/numpy/ann_numpy.py at master · animesh/ANN · GitHub](https://github.com/animesh/ANN/blob/master/numpy/ann_numpy.py) 🤞
