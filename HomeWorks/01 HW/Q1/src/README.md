# DFA Simulation with McCulloch–Pitts Neurons

> **Pattern:** detect the binary substring `100`.  
> **Idea:** implement a **Deterministic Finite Automaton (DFA)** and reproduce its behavior with a tiny network of **McCulloch–Pitts** (threshold) neurons using integer weights and a shared threshold.


---

## Overview
This repository implements a minimal DFA that accepts a binary input if it **contains** the pattern `1 -> 0 -> 0`.  
Once the pattern is seen, the DFA moves to an **accepting** state and **stays there** for the rest of the input.

We also simulate the same logic with a very small network of McCulloch–Pitts neurons to show how simple neural units can replicate finite-state behavior.

> **Educational use:** Great for courses in *Automata Theory*, *Intro to Neural Networks*, or *Symbolic/Logical Computation*.

---

## DFA Description
- **Alphabet:** `{0, 1}`  
- **States:** `0` (start), `1` and `2` (intermediate), `3` (accepting)  
- **Behavior:** start at `0`, look for `1, 0, 0`. After seeing that sequence, transition to `3` and remain there.

### Binary State Encoding (for the neural model)
We encode each state with two bits:
```
0 ↔ 00
1 ↔ 01
2 ↔ 10
3 ↔ 11
```
So the **neuron inputs** at each step are: `s1`, `s0` (current-state bits) and `x` (current input bit).  
The **neuron outputs** are: `s1'`, `s0'` (next-state bits) and `acc` (1 if the next state is accepting, else 0).

---

## State Transition Table
| Current | Input | Next | Accept |
|:------:|:----:|:----:|:-----:|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 2 | 0 |
| 1 | 1 | 1 | 0 |
| 2 | 0 | 3 | 1 |
| 2 | 1 | 1 | 0 |
| 3 | 0 | 3 | 1 |
| 3 | 1 | 3 | 1 |

> In state `3`, the DFA locks in acceptance.

---

## Neural Simulation
We model each output bit (`s1'`, `s0'`, `acc`) with an **independent McCulloch–Pitts neuron** sharing the same threshold `θ` and using integer weights.

**Neuron rule:**  
If `wᵀx ≥ θ`, output `1`; otherwise `0`.

In this project:
- input vector: `x = [s1, s0, in]`
- shared threshold: `θ = 2`
- weights:
  - for `s1'`: `w = [2, 2, -1]`
  - for `s0'`: `w = [2, 0, 2]`
  - for `acc`:  `w = [2, 1, -1]`

These weights reproduce the transition table and acceptance logic exactly.

---

## Project Structure
```
.
├── README.md
└── src/Q1.ipynb
```

---

## Installation
**Requirements**
- Python 3.9+
- NumPy

**Install dependencies**
```bash
pip install numpy
```

---

## Example Run
Input: `011001`  
Steps:
1. start in `0` (`00`)
2. read `0` -> stay in `0`
3. read `1` -> go to `1`
4. read `1` -> stay in `1`
5. read `0` -> go to `2`
6. read `0` -> go to `3` *(accepting)*
7. read `1` -> stay in `3` *(accepting)*

The string is accepted.

---

## Validation
To sanity-check the neuron logic against all `(state, input)` pairs:

```python
import itertools, numpy as np

def DFA_once(state_bits, input_bit):
    inputs = np.array([state_bits[0], state_bits[1], input_bit])
    s1p = McCullochPittsNeuron([2, 2, -1], 2)(inputs)
    s0p = McCullochPittsNeuron([2, 0, 2], 2)(inputs)
    acc = McCullochPittsNeuron([2, 1, -1], 2)(inputs)
    return (s1p, s0p, acc)

states = [(0,0),(0,1),(1,0),(1,1)]  # 00,01,10,11  <->  0,1,2,3
for s in states:
    for x in [0,1]:
        print("state:", s, "input:", x, "->", DFA_once(s, x))
```
Expected outputs match the transition table; `acc = 1` when entering or being in state `3`.

---

## Design Notes
- Using **integer weights** and a **shared threshold** (`θ = 2`) keeps the network interpretable as a simple threshold logic circuit.
- The `acc` neuron fires on the `2 + 0 -> 3` transition and remains on in state `3`, mirroring the DFA’s absorbing acceptance.
- This illustrates how a tiny threshold network can encode a DFA; more complex patterns would require additional neurons/layers.

> This is an **educational** simulation intended to illustrate DFA ↔ threshold-network mapping, not to compete with deep learning models.

---

## License
Open for **academic use**. Attribution is appreciated.

## Contact
Questions or collaboration ideas? Open an issue or reach out directly.
