import math
import random

def transpose(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

def matmul(m1, m2):
    m2_t = transpose(m2)
    return [[dot_product(row, col) for col in m2_t] for row in m1]

def add(m1, m2):
    return [[a + b for a, b in zip(row1, row2)] for row1, row2 in zip(m1, m2)]

def sub(m1, m2):
    return [[a - b for a, b in zip(row1, row2)] for row1, row2 in zip(m1, m2)]

def mul_scalar(s, m):
    return [[x * s for x in row] for row in m]

def sum_axis_0(m):
    cols = len(m[0])
    res = [0.0] * cols
    for row in m:
        for i in range(cols):
            res[i] += row[i]
    return [res]

class Layer:
    def forward(self, input_data): raise NotImplementedError
    def backward(self, output_error, learning_rate): raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.w = [[random.random() * 0.5 for _ in range(output_size)] for _ in range(input_size)]
        self.b = [[0.0 for _ in range(output_size)]]
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        xw = matmul(self.input, self.w)
        return [[xw_row[i] + self.b[0][i] for i in range(len(self.b[0]))] for xw_row in xw]

    def backward(self, output_error, learning_rate):
        input_error = matmul(output_error, transpose(self.w))
        weights_grad = matmul(transpose(self.input), output_error)
        bias_grad = sum_axis_0(output_error)

        self.w = sub(self.w, mul_scalar(learning_rate, weights_grad))
        self.b = sub(self.b, mul_scalar(learning_rate, bias_grad))
        return input_error

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return [[max(0.0, x) for x in row] for row in input_data]

    def backward(self, output_error, learning_rate):
        return [[e * (1.0 if x > 0.0 else 0.0) for e, x in zip(e_row, x_row)] 
                for e_row, x_row in zip(output_error, self.input)]

class Softmax(Layer):
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        out = []
        for row in input_data:
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            out.append([e / sum_exps for e in exps])
        self.output = out
        return self.output

    def backward(self, output_error, learning_rate):
        return output_error

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad, lr):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad, lr)

    def train(self, x, y, batch_size, epochs, lr):
        n_samples = len(x)
        for epoch in range(1, epochs + 1):
            for i in range(0, n_samples, batch_size):
                bx = x[i:i + batch_size]
                by = y[i:i + batch_size]
                
                pred = self.forward(bx)
                n = len(bx)
                loss_grad = [[(p - t) / n for p, t in zip(p_row, t_row)] for p_row, t_row in zip(pred, by)]
                
                self.backward(loss_grad, lr)
                
            preds = self.forward(x)
            loss = self.cross_entropy(preds, y)
            print(f"Epoch {epoch}, Loss: {loss:.6f}", flush=True)

    @staticmethod
    def cross_entropy(pred, target):
        epsilon = 1e-15
        total_loss = 0.0
        n = len(pred)
        for p_row, t_row in zip(pred, target):
            for p, t in zip(p_row, t_row):
                p_clip = max(epsilon, min(1.0 - epsilon, p))
                total_loss += -(t * math.log(p_clip))
        return total_loss / n
