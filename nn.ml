open Matrix
module M = Matrix

(* between layers we have weights and biases *)
type layer = {
        w: matrix; (* n_l * n_(l+1) matrix *)
        b: matrix; (* 1 x  N matrix *)
}

(* a model is a list of layers *)
type model = layer list

(** Layer Initialization **)
let init_layer rows cols =
  (* Random.float 1.0 -. 0.5 returns [-0.5, 0.5) *)
  let w = List.init rows (fun _ -> List.init cols (fun _ -> Random.float 1.0 -. 0.5)) in
  let b = List.init 1 (fun _ -> List.init cols (fun _ -> 0.0)) in
  { w = w; b = b }

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
(* Here target must be a one hot vector *)
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
let forward model input = 
        let rec help layers act_acc z_acc = 
                match layers with 
                | [] -> (act_acc, z_acc)
                | layer:: rest -> 
                  let xw = matmul (List.hd act_acc) layer.w in
                  let xw_plus_b = List.map (fun row -> List.hd (add [row] layer.b)) xw in
                  let next_act = if rest = [] then softmax xw_plus_b else relu xw_plus_b in
                  help rest (next_act :: act_acc) (xw_plus_b::z_acc)
        in help model [input] [] 

(** Backward Pass **)
(* returns a list of layers representing the gradients (same shape as model) *)
(* here dvar = dL / dvar *)
let backward model acts zs target_y =
  let batch_size_f = float_of_int (List.length target_y) in
  let final_pred = List.hd acts in
  
  (* For softmax + categorical cross entropy, dZ_L = (Pred - Target) / batch_size *)
  let dz_l = map2 (fun p t -> (p -. t) /. batch_size_f) final_pred target_y in
  
  let rec help rev_model acts_tl current_zs current_dz acc_grads =
    match rev_model, acts_tl, current_zs with
    | [], _, _ -> acc_grads
    | layer :: rest_model, a_prev :: rest_acts, z :: rest_zs ->
        (* dw = a^T dz *)            
        let dw = matmul (transpose a_prev) current_dz in
        (* db = sum over i dz_i *)
        let db = sum_axis_0 current_dz in
        (* the current layer's dL/dw and dL/db *)
        let layer_grad = { w = dw; b = db } in
        
        if rest_model = [] then
          layer_grad :: acc_grads
        else
          (* da_(l-1) = dz_l * (w_l)^T *)
          let da_prev = matmul current_dz (transpose layer.w) in
          let z_prev = List.hd rest_zs in
          (* dz_(l-1) = da_(l-1) elwise prod relu'(z_(l-1)) *)
          let dz_prev = mul_elementwise da_prev (relu_derivative z_prev) in
          (* pass the new dz i.e. the previous layer's dz and the updated acc_grads *)
          help rest_model rest_acts rest_zs dz_prev (layer_grad :: acc_grads)
    | _ -> failwith "Mismatched shapes in backward lists"
  in
  (* reverse the model because forwarad returns acts and zs reversed *)
  help (List.rev model) (List.tl acts) zs dz_l []


(** Update Layer & Model **)
 (* w = w - lr * dw *)
 (* b = b - lr * db *)
let update_layer lr layer grad =
  { w = sub layer.w (mul_scalar lr grad.w);
    b = sub layer.b (mul_scalar lr grad.b) }

let update_model lr model grads =
  List.map2 (update_layer lr) model grads

(** Helper to split list into chunks **)
let rec chunks l n =
  (* current is the current batch *)
  (* i is the number of elements in the batch *)
  let rec aux acc current i lst =
    match lst with
    | [] -> if current = [] then List.rev acc else List.rev (List.rev current :: acc)
    | h :: t ->
        if i = n then aux (List.rev current :: acc) [h] 1 t
        else aux acc (h :: current) (i + 1) t
  in
  aux [] [] 0 l

(** Training Loop with mini-batches **)
let train model x y batch_size epochs lr =
  let x_batches = chunks x batch_size in
  let y_batches = chunks y batch_size in
  let batches = List.combine x_batches y_batches in
  
  let rec loop current_model epoch =
    if epoch = 0 then current_model
    else
    (* use cur_model and a batch of data to get the updated model *)
      let new_model = List.fold_left (fun cur_mod (bx, by) ->
        let (acts, zs) = forward cur_mod bx in
        let grads = backward cur_mod acts zs by in
        update_model lr cur_mod grads
      ) current_model batches in
    (* inference with final model *)  
      let (acts, _) = forward new_model x in
      let loss = cross_entropy (List.hd acts) y in
      if true then Printf.printf "Epoch %d, Loss: %f\n%!" epoch loss;
      loop new_model (epoch - 1)
  in
  loop model epochs
