open Matrix

val sigmoid : matrix -> matrix
val relu : matrix -> matrix
val tanh : matrix -> matrix

val sigmoid_derivative : matrix -> matrix
val relu_derivative : matrix -> matrix
val tanh_derivative : matrix -> matrix

val mse : matrix -> matrix -> float
val mse_derivative : matrix -> matrix -> matrix

val bce : matrix -> matrix -> float
val bce_derivative : matrix -> matrix -> matrix
