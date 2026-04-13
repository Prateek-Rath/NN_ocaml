type layer

type model

val sigmoid : Matrix.matrix -> Matrix.matrix
val softmax: Matrix.matrix -> Matrix.matrix
val relu : Matrix.matrix -> Matrix.matrix
val tanh : Matrix.matrix -> Matrix.matrix

val sigmoid_derivative : Matrix.matrix -> Matrix.matrix
val relu_derivative : Matrix.matrix -> Matrix.matrix
val tanh_derivative : Matrix.matrix -> Matrix.matrix

val mse : Matrix.matrix -> Matrix.matrix -> float
val mse_derivative : Matrix.matrix -> Matrix.matrix -> Matrix.matrix

val bce : Matrix.matrix -> Matrix.matrix -> float
val bce_derivative : Matrix.matrix -> Matrix.matrix -> Matrix.matrix

val cross_entropy : Matrix.matrix -> Matrix.matrix -> float
val cross_entropy_derivative : Matrix.matrix -> Matrix.matrix -> Matrix.matrix

