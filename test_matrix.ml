open Matrix

let print_matrix m =
  List.iter (fun row ->
    List.iter (fun x -> Printf.printf "%.2f " x) row;
    print_newline ()
  ) m

let () =
  let m1 = [[1.; 2.]; [3.; 4.]] in
  let m2 = [[5.; 6.]; [7.; 8.]] in
  
  Printf.printf "Matrix M1:\n";
  print_matrix m1;
  
  Printf.printf "Shape of M1: (%d, %d)\n" (fst (shape m1)) (snd (shape m1));
  
  Printf.printf "M1 + M2:\n";
  print_matrix (add m1 m2);
  
  Printf.printf "M1 * M2 (matmul):\n";
  print_matrix (matmul m1 m2);
  
  Printf.printf "M1 element-wise * M2:\n";
  print_matrix (mul_elementwise m1 m2);
  
  let s0 = sum_axis_0 m1 in
  Printf.printf "Sum axis 0 of M1:\n";
  print_matrix s0;
  
  let flattened = flatten m1 in
  Printf.printf "Flattened M1: [";
  List.iter (fun x -> Printf.printf "%.1f " x) flattened;
  Printf.printf "]\n";
  
  let m_zeros = make 2 3 0.0 in
  Printf.printf "2x3 Zeros:\n";
  print_matrix m_zeros;
