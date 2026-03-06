"""Mixed-precision accumulation exercise (§1.1.5): run and comment on the results."""

import torch

# (a) FP32 accumulator + FP32 addends → exact 10.0
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

# (b) FP16 accumulator + FP16 addends → loss of precision
s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

# (c) FP32 accumulator + FP16 addends (no cast) → implicit cast, still can lose
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

# (d) FP32 accumulator + FP16 addends cast to FP32 before add → exact 10.0
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.to(torch.float32)
print(s)