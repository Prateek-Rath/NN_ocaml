type layer

type model = layer list

val init_layer: int -> int -> layer

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

val forward: model -> Matrix.matrix -> Matrix.matrix list * Matrix.matrix list

val backward: model -> Matrix.matrix list -> Matrix.matrix list -> Matrix.matrix -> model
val update_layer: float -> layer -> layer -> layer
val update_model: float -> model -> model -> model
val train: model -> Matrix.matrix -> Matrix.matrix -> int -> int -> float -> model
