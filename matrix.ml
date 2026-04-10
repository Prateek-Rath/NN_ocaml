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
