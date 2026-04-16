open Matrix
open Nn

(* --- CSV Parsing & Preprocessing (Pure Functional) --- *)

let map_gender g = match g with "female" -> 1.0 | "male" -> 0.0 | _ -> 0.0
let map_education e = match e with "Master" -> 4.0 | "Doctorate" -> 3.0 | "Bachelor" -> 2.0 | "Associate" -> 1.0 | "High School" -> 0.0 | _ -> 0.0
let map_ownership o = match o with "MORTGAGE" -> 3.0 | "OWN" -> 2.0 | "RENT" -> 1.0 | "OTHER" -> 0.0 | _ -> 0.0
let map_intent i = match i with "PERSONAL" -> 0.0 | "EDUCATION" -> 1.0 | "MEDICAL" -> 2.0 | "VENTURE" -> 3.0 | "HOMEIMPROVEMENT" -> 4.0 | "DEBTCONSOLIDATION" -> 5.0 | _ -> 0.0
let map_defaults d = match d with "Yes" -> 1.0 | "No" -> 0.0 | _ -> 0.0

let parse_float s = try float_of_string s with _ -> 0.0

let parse_line line =
  let cols = String.split_on_char ',' line in
  match cols with
  | [age; gender; ed; inc; emp; own; amnt; intent; int_r; perc_inc; cred_l; cred_s; def; status] ->
      let features = [
        parse_float age; map_gender gender; map_education ed; parse_float inc;
        parse_float emp; map_ownership own; parse_float amnt; map_intent intent;
        parse_float int_r; parse_float perc_inc; parse_float cred_l; parse_float cred_s;
        map_defaults def
      ] in
      let label = match status with "1" -> [0.0; 1.0] | _ -> [1.0; 0.0] in
      Some (features, label)
  | _ -> None

let read_dataset filename max_samples =
  let ic = open_in filename in
  let rec loop acc_x acc_y header_skipped count =
    if count >= max_samples then (
      close_in ic;
      (List.rev acc_x, List.rev acc_y)
    ) else
      try
        let line = input_line ic in
        let trimmed = String.trim line in
        if not header_skipped then loop acc_x acc_y true count
        else if trimmed = "" then loop acc_x acc_y true count
        else
          match parse_line trimmed with
          | Some (x, y) -> loop (x :: acc_x) (y :: acc_y) true (count + 1)
          | None -> loop acc_x acc_y true count
      with End_of_file ->
        close_in ic;
        (List.rev acc_x, List.rev acc_y)
  in
  loop [] [] false 0

(* functional min-max scaler *)
let min_max_scale m =
  let min_vals = List.hd (Matrix.min_axis_0 m) in
  let max_vals = List.hd (Matrix.max_axis_0 m) in
  let diffs = List.map2 (fun mx mn -> let d = mx -. mn in if d = 0.0 then 1.0 else d) max_vals min_vals in
  Matrix.map2 (fun row_el idx -> 
      (* we can't use map2 easily without index, so let's write a small helper *)
      0.0
    ) m m (* won't work perfectly that way, let's just map explicitly *)
  |> ignore;
  
  List.map (fun row ->
    let rec scale_row r mins diffs_l =
      match r, mins, diffs_l with
      | [], _, _ -> []
      | h::t, mn::mnt, d::dt -> ((h -. mn) /. d) :: scale_row t mnt dt
      | _ -> []
    in
    scale_row row min_vals diffs
  ) m


let () =
  Random.self_init ();
  Printf.printf "Loading dataset (full)...\n%!";
  let (raw_x, y) = read_dataset "loan_data.csv" 500000 in
  Printf.printf "Loaded %d samples.\n%!" (List.length raw_x);
  
  (* scale X *)
  let x = min_max_scale raw_x in
  
  (* let's build a model: 13 features -> 16 hidden -> 2 output *)
  let model = [
    Nn.init_layer 13 16;
    Nn.init_layer 16 2
  ] in
  
  (* shuffle data *)
  let shuffle_data x_lst y_lst =
    let pairs = List.combine x_lst y_lst in
    let tagged = List.map (fun p -> (Random.bits (), p)) pairs in
    let sorted = List.sort compare tagged in
    let shuffled_pairs = List.map snd sorted in
    List.split shuffled_pairs
  in
  let (shuffled_x, shuffled_y) = shuffle_data x y in
  
  Printf.printf "Starting training...\n%!";
  let start_time = Sys.time () in
  let epochs = 20 in
  let batch_size = 32 in
  let lr = 0.01 in
  let final_model = Nn.train model shuffled_x shuffled_y batch_size epochs lr in
  let end_time = Sys.time () in
  
  (* just do one forward pass to check final loss and accuracy *)
  let (acts, _) = Nn.forward final_model x in
  let preds = List.hd acts in
  let final_loss = Nn.cross_entropy preds y in
  Printf.printf "Final model loss: %f\n" final_loss;
  
  (* Calculate accuracy *)
  let correct = List.fold_left2 (fun acc p t ->
    let p0 = List.hd p in
    let p1 = List.hd (List.tl p) in
    let is_class_1 = p1 > p0 in
    let is_target_1 = (List.hd (List.tl t)) = 1.0 in
    if is_class_1 = is_target_1 then acc + 1 else acc
  ) 0 preds y in
  let total = List.length y in
  let accuracy = float_of_int correct /. float_of_int total *. 100.0 in
  Printf.printf "Accuracy: %.2f%%\n" accuracy;
  Printf.printf "Time taken: %.2f seconds\n" (end_time -. start_time)

