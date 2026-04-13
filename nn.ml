open Matrix
module M = Matrix

(* between layers we have weights and biases *)
type layer = {
        w: matrix; (* n_l * n_(l+1) matrix *)
        b: matrix; (* 1 x  N matrix *)
}

(* a model is a list of layers *)
type model = layer list


(** Activation Functions **)

let sigmoid_fn x = 1.0 /. (1.0 +. exp (-.x))

let sigmoid m = map sigmoid_fn m

let relu_fn x = if x > 0.0 then x else 0.0

let relu m = map relu_fn m

let tanh m = map (fun x -> tanh x) m

let softmax_vector v =
        let max_val = List.fold_left max (-1e9) v in 
        let exps = List.map (fun x -> exp (x -. max_val)) v in (* this prevents div by very small number *)
        let sum_exps = List.fold_left (+.) 0.0 exps in 
        List.map (fun x -> x /. sum_exps) exps 

let softmax m = 
        List.map (fun row -> softmax_vector row) m


(** Derivatives **)

(* sigmoid_derivative expects the output of the sigmoid function (post-activation) *)
let sigmoid_derivative post_act = 
  map (fun s -> s *. (1.0 -. s)) post_act

let relu_derivative pre_act =
  map (fun x -> if x > 0.0 then 1.0 else 0.0) pre_act

(* tanh_derivative expects the output of the tanh function (post-activation) *)
let tanh_derivative post_act =
  map (fun t -> 1.0 -. (t *. t)) post_act

(* we won't need softmax derivative *)
(* 

        while using softmax + cross-entropy, the derivative simplifies to:
        gradient = pred − target

*)

(** Loss Functions **)

(* Mean Squared Error: returns a single float scalar (wrapped in a matrix or list if needed, 
   but usually we want the mean of all elements) *)
let mse pred target =
  let diffSq = map2 (fun p t -> (p -. t) ** 2.0) pred target in
  let flat = flatten diffSq in
  let sum = List.fold_left (+.) 0.0 flat in
  sum /. (float_of_int (List.length flat))

let mse_derivative pred target =
  let n = float_of_int (List.length (flatten pred)) in
  map2 (fun p t -> (2.0 /. n) *. (p -. t)) pred target

(* Binary Cross Entropy: returns a single float scalar *)
let bce pred target =
  let epsilon = 1e-15 in (* to avoid log(0) *)
  let losses = map2 (fun p t ->
    let p = if p < epsilon then epsilon else if p > 1.0 -. epsilon then 1.0 -. epsilon else p in
    -. (t *. log p +. (1.0 -. t) *. log (1.0 -. p))
  ) pred target in
  let flat = flatten losses in
  let sum = List.fold_left (+.) 0.0 flat in
  sum /. (float_of_int (List.length flat))

let bce_derivative pred target =
  let epsilon = 1e-15 in
  let n = float_of_int (List.length (flatten pred)) in
  map2 (fun p t ->
    let p = if p < epsilon then epsilon else if p > 1.0 -. epsilon then 1.0 -. epsilon else p in
    (1.0 /. n) *. (p -. t) /. (p *. (1.0 -. p))
  ) pred target
  
(* Categorical Cross Entropy: returns a single float scalar *)
let cross_entropy pred target =
  let epsilon = 1e-15 in
  let losses = map2 (fun p t ->
    let p =
      if p < epsilon then epsilon
      else if p > 1.0 -. epsilon then 1.0 -. epsilon
      else p
    in
    -. (t *. log p)
  ) pred target in
  let flat = flatten losses in
  let sum = List.fold_left (+.) 0.0 flat in
  sum /. (float_of_int (List.length pred))

let cross_entropy_derivative pred target =
  let n = float_of_int (List.length pred) in
  map2 (fun p t ->
    (p -. t) /. n
  ) pred target


(** Forward Pass **)
(* take in model i.e a list of layers and input *)
(* returns (activations, z_values) in reverse order for backprop *)
        
