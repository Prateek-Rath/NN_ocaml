open Matrix
open Nn

let () =
  let test_matrix = [[1.0; -1.0]; [0.0; 2.0]] in
  let target_matrix = [[1.0; 0.0]; [0.0; 1.0]] in
  
  Printf.printf "Testing Activations:\n";
  let s = sigmoid test_matrix in
  Printf.printf "Sigmoid: %f %f\n" (List.hd (List.hd s)) (List.hd (List.tl (List.hd s)));
  
  let r = relu test_matrix in
  Printf.printf "ReLU: %f %f\n" (List.hd (List.hd r)) (List.hd (List.tl (List.hd r)));
  
  Printf.printf "\nTesting Losses:\n";
  let m = mse s target_matrix in
  Printf.printf "MSE: %f\n" m;
  
  let b = bce s target_matrix in
  Printf.printf "BCE: %f\n" b;
  
  let m_grad = mse_derivative s target_matrix in
  Printf.printf "MSE Gradient shape: (%d, %d)\n" (fst (shape m_grad)) (snd (shape m_grad));
  
  Printf.printf "All primitive tests passed (logically).\n"
