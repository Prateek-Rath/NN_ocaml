type matrix = float list list;;

type vector = float list;;

(* matrix transpose *)
let rec transpose m = match m with
        [] -> [] |
        []::_ -> [] | (* if the first list is empty then transpose is empty*)
        rows -> List.map List.hd rows :: transpose (List.map List.tl rows)

(* dot product *)
let rec dot_product v1 v2 =
        let rec get_prod_list v1 v2 =
                match v1 with
                [] -> [] |
                h1::t1 -> match v2 with [] -> [] | h2::t2 -> h1*.h2 :: (get_prod_list t1 t2)
        in List.fold_left (+.) 0. (get_prod_list v1 v2);;

(* matrix vector multiplication *)
let rec matr_vec_mul m v = 
        match m with 
        | [] -> [] 
        | h::t -> (dot_product h v):: matr_vec_mul t v;;


(* normal matrix multiplication *)
let matrmul m1 m2 = 
        let m2_t = transpose m2 in 
        let rec help x y = match y with 
                                [] -> [] |
                                h2::t2 -> matr_vec_mul x h2 :: help x t2 
        in let tmp = help m1 m2_t
        in transpose tmp;;

(** Element-wise mapping **)
let map f m = List.map (List.map f) m

(** Element-wise mapping with two matrices **)
let map2 f m1 m2 =
  List.map2 (List.map2 f) m1 m2

(** Matrix addition (standard) **)
let add = map2 (+.)

(** Matrix subtraction **)
let sub = map2 (-.)

(** Scalar multiplication **)
let mul_scalar s = map (fun x -> x *. s)

(** Sum along axis 0 (columns) **)
let sum_axis_0 m =
  match m with
  | [] -> []
  | first :: _ ->
      let cols = List.length first in
      let init = List.init cols (fun _ -> 0.0) in
      [List.fold_left (fun acc row -> List.map2 (+.) acc row) init m]

(** Max along axis 0 (columns) **)
let max_axis_0 m =
  match m with
  | [] -> []
  | first :: _ ->
      let cols = List.length first in
      let init = List.init cols (fun _ -> -.infinity) in
      [List.fold_left (fun acc row -> List.map2 max acc row) init m]

(** Min along axis 0 (columns) **)
let min_axis_0 m =
  match m with
  | [] -> []
  | first :: _ ->
      let cols = List.length first in
      let init = List.init cols (fun _ -> infinity) in
      [List.fold_left (fun acc row -> List.map2 min acc row) init m]

(** Element-wise product **)
let mul_elementwise = map2 ( *. )

(** Matrix multiplication: A x B **)
let matmul a b =
  let b_t = transpose b in
  List.map (fun row -> List.map (fun col -> dot_product row col) b_t) a

(** Helper to create a matrix with a single value **)
let make rows cols v =
  List.init rows (fun _ -> List.init cols (fun _ -> v))

(** Flatten a matrix (batch) into a single list of floats (if needed) **)
let flatten m = List.concat m

(** Shape of a matrix (rows, cols) **)
let shape m =
  match m with
  | [] -> (0, 0)
  | row :: _ -> (List.length m, List.length row)
