import math

class Value:
    def __init__(self, data, _prev: tuple[Value]=(), _operation=''):
        self.data = float(data)
        self._prev = _prev
        self.grad = 0.0
        self._operation = _operation

        self._backward = lambda: None

    def __repr__(self):
        return f"Value:{self.data} ; grad: {self.grad}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data, 
            (self, other), 
            _operation='+'
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data*other.data, 
            (self, other), 
            _operation='*'
        )

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out
    __rmul__=__mul__
    # def build_computational_graph(self):
    #     computational_graph = []

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data**other.data, 
            (self, other), 
            _operation='**'
        )
        def _backward():
            self.grad += other.data * (self.data**(other.data-1))
            other.grad += out.data * math.log(self.data, math.e)

        out._backward = _backward
        return out

    def build_topo(self) -> list[Value]:
        topo = list()
        visited = set()

        def build(node: Value):
            if node not in visited:
                visited.add(node)
                #   We put this loop before topo.append(node), 
                #   so it checks if there is previous node first
                for prev in node._prev:
                    build(prev)
                topo.append(node)
        build(self)
        return reversed(topo)

    def backward(self):
        self.grad += 1.0
        reversed_topo = self.build_topo()
        for node in reversed_topo:
            node._backward()


if __name__ == '__main__':
    x = Value(3)
    a = x * x
    b = a * 2
    c = a * 3

    y = b + c

    y.backward()
    print(x)