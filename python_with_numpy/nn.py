import numpy as np

class Layer:
    def forward(self, input_data):
        raise NotImplementedError

    def backward(self, output_error, learning_rate):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Match OCaml Random.float 0.5 behavior
        self.w = np.random.rand(input_size, output_size) * 0.5
        self.b = np.zeros((1, output_size))
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.dot(self.input, self.w) + self.b

    def backward(self, output_error, learning_rate):
        # input_error = dL/dX = dL/dY * W^T
        input_error = np.dot(output_error, self.w.T)
        
        # weights_grad = dL/dW = X^T * dL/dY
        weights_grad = np.dot(self.input.T, output_error)
        
        # bias_grad = dL/dB = sum(dL/dY)
        bias_grad = np.sum(output_error, axis=0, keepdims=True)

        # Update parameters
        self.w -= learning_rate * weights_grad
        self.b -= learning_rate * bias_grad

        return input_error

class ReLU(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, output_error, learning_rate):
        # learning_rate is not used for activation layer
        return output_error * (self.input > 0).astype(float)

class Softmax(Layer):
    def __init__(self):
        self.output = None

    def forward(self, input_data):
        max_val = np.max(input_data, axis=1, keepdims=True)
        exps = np.exp(input_data - max_val)
        self.output = exps / np.sum(exps, axis=1, keepdims=True)
        return self.output

    def backward(self, output_error, learning_rate):
        # Since we use Categorical Cross Entropy + Softmax,
        # the derivative calculation simplifies and we assume the `output_error` 
        # passed here is already the gradient w.r.t the logits (Z).
        # Therefore, we just pass the gradient backwards.
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
        n_samples = x.shape[0]

        for epoch in range(1, epochs + 1):
            # To match OCaml mini-batch exactly, we just iterate sequentially
            for i in range(0, n_samples, batch_size):
                bx = x[i:i + batch_size]
                by = y[i:i + batch_size]

                # Forward Pass
                pred = self.forward(bx)

                # Cross-Entropy + Softmax Derivative w.r.t Z: (pred - target) / N
                n = bx.shape[0]
                loss_grad = (pred - by) / n

                # Backward Pass
                self.backward(loss_grad, lr)

            # Evaluate full dataset at the end of epoch
            preds = self.forward(x)
            loss = self.cross_entropy(preds, y)
            print(f"Epoch {epoch}, Loss: {loss:.6f}", flush=True)

    @staticmethod
    def cross_entropy(pred, target):
        epsilon = 1e-15
        pred = np.clip(pred, epsilon, 1.0 - epsilon)
        losses = -(target * np.log(pred))
        return np.sum(losses) / pred.shape[0]
