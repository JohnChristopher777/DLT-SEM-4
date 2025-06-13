def mc_neuron(inputs, weights, threshold):
    summation = sum(i * w for i, w in zip(inputs, weights))
    return 1 if summation >= threshold else 0

def AND(x1, x2):
    return mc_neuron([x1, x2], [1, 1], 2)

def OR(x1, x2):
    return mc_neuron([x1, x2], [1, 1], 1)

def NOT(x):
    return mc_neuron([x], [-1], 0)

def NOR(x1, x2):
    return mc_neuron([x1, x2], [-1, -1], -0.5)

def XOR(x1, x2):
    return x1 ^ x2

def NAND(x1, x2):
    return mc_neuron([x1, x2], [-1, -1], -1.5)

print("AND")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) -> {AND(x1,x2)}")

print("\nOR")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) -> {OR(x1,x2)}")

print("\nNOR")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) -> {NOR(x1,x2)}")

print("\nXOR")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) -> {XOR(x1,x2)}")

print("\nNOT")
for x in [0, 1]:
    print(f"({x}) -> {NOT(x)}")

print("\nNAND")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(f"({x1},{x2}) -> {NAND(x1,x2)}")