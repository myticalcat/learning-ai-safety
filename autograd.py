"""
Paper 1: Backpropagation (Rumelhart, Hinton & Williams, 1986)
Autograd engine mini, dibangun dari nol dengan Python murni.
"""
import math


class Value:
    """Satu angka skalar yang 'ingat' bagaimana ia dihitung, sehingga bisa backprop."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0                 # dL/d(node ini), awalnya 0
        self._backward = lambda: None   # cara meneruskan gradien ke anak-anaknya
        self._prev = set(_children)     # node yang membentuk node ini
        self._op = _op                  # label operasi (untuk debugging)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ---------- Operasi dasar ----------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # turunan lokal penjumlahan = 1, jadi gradien diteruskan apa adanya
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad -= out.grad
            other.grad -= out.grad 
        out._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data ** out.grad
            other.grad += self.data ** out.grad
        out._backward = _backward
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # turunan lokal perkalian = nilai "pasangannya"
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # supaya 2 + Value(3) dan 2 * Value(3) juga bisa
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # ---------- Backpropagation ----------

    def backward(self):
        # 1. Urutkan semua node secara topologis (anak selalu sebelum induk)
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        # 2. dL/dL = 1, lalu jalan mundur dan terapkan chain rule di setiap node
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# =====================================================================
# Tes 1: bandingkan dengan perhitungan manual
# =====================================================================
def forward(a_val, b_val, c_val, f_val):
    a, b, c, f = Value(a_val), Value(b_val), Value(c_val), Value(f_val)
    e = a * b
    d = e - c
    L = d ** f
    return L, (a, b, c, f, e, d)


L, (a, b, c, f, e, d) = forward(2.0, -3.0, 10.0, -2.0)
L.backward()

print("=== Hasil autograd ===")
print("L =", L)
for name, v in zip("abcfed", (a, b, c, f, e, d)):
    print(f"dL/d{name} = {v.grad}")
# Harapan: dL/da=6, dL/db=-4, dL/dc=-2, dL/df=4, dL/de=-2, dL/dd=-2

# =====================================================================
# Tes 2: numerical gradient check  (f(x+h) - f(x)) / h
# =====================================================================
print("\n=== Numerical gradient check ===")
h = 1e-6
base = forward(2.0, -3.0, 10.0, -2.0)[0].data
inputs = {"a": 2.0, "b": -3.0, "c": 10.0, "f": -2.0}
for name in inputs:
    bumped = dict(inputs)
    bumped[name] += h
    numerical = (forward(bumped["a"], bumped["b"], bumped["c"], bumped["f"])[0].data - base) / h
    print(f"dL/d{name} numerik ≈ {numerical:.4f}")

# =====================================================================
# Tes 3: kenapa harus += ?  (variabel dipakai dua kali)
# =====================================================================
print("\n=== Variabel dipakai dua kali ===")
x = Value(3.0)
y = x + x          # dy/dx seharusnya 2
y.backward()
print("dy/dx =", x.grad, "(harusnya 2.0)")