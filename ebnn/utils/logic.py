import chainer.functions as F

def f_not(a):
    return 1.0 - a

def f_and(a, b):
    # Product T-norm
    return a * b

def f_or(a, b):
    # Product T-conorm
    return a + b - a * b

def f_xor(a, b):
    return a * (1.0 - b) + b * (1.0 - a)

def f_implies(a, b):
    # Reichenbach implication: 1 - a + ab
    return 1.0 - a + a * b

def f_equivalent(a, b):
    # 1 - |a - b|
    return 1.0 - F.absolute(a - b)

def f_nand(a, b):
    return f_not(f_and(a, b))

def f_nor(a, b):
    return f_not(f_or(a, b))
