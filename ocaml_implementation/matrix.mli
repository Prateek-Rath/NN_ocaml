type matrix = float list list
type vector = float list

val transpose: matrix -> matrix
val dot_product: vector -> vector -> float
val matr_vec_mul: matrix -> vector -> vector
val matmul: matrix -> matrix -> matrix

val map: (float -> float) -> matrix -> matrix
val map2: (float -> float -> float) -> matrix -> matrix -> matrix
val add: matrix -> matrix -> matrix
val sub: matrix -> matrix -> matrix
val mul_scalar: float -> matrix -> matrix
val sum_axis_0: matrix -> matrix
val max_axis_0: matrix -> matrix
val min_axis_0: matrix -> matrix
val mul_elementwise: matrix -> matrix -> matrix
val make: int -> int -> float -> matrix
val flatten: matrix -> float list
val shape: matrix -> int * int
