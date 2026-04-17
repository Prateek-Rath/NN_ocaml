# Compiler and flags

OCAML = ocamlc
FLAGS = -g

# Default target

all: test_matrix test_nn train_loan

# ---- Interface files (.mli → .cmi) ----

matrix.cmi: matrix.mli
	$(OCAML) $(FLAGS) -c matrix.mli

nn.cmi: nn.mli matrix.cmi
	$(OCAML) $(FLAGS) -c nn.mli

train_loan.cmi: train_loan.mli matrix.cmi nn.cmi
	$(OCAML) $(FLAGS) -c train_loan.mli

# ---- Implementation files (.ml → .cmo) ----

matrix.cmo: matrix.ml matrix.cmi
	$(OCAML) $(FLAGS) -c matrix.ml

nn.cmo: nn.ml nn.cmi matrix.cmi
	$(OCAML) $(FLAGS) -c nn.ml

test_matrix.cmo: test_matrix.ml matrix.cmi
	$(OCAML) $(FLAGS) -c test_matrix.ml

test_nn.cmo: test_nn.ml matrix.cmi nn.cmi
	$(OCAML) $(FLAGS) -c test_nn.ml

train_loan.cmo: train_loan.ml matrix.cmi nn.cmi
	$(OCAML) $(FLAGS) -c train_loan.ml

# ---- Linking executables ----

test_matrix: matrix.cmo test_matrix.cmo
	$(OCAML) -o test_matrix matrix.cmo test_matrix.cmo

test_nn: matrix.cmo nn.cmo test_nn.cmo
	$(OCAML) -o test_nn matrix.cmo nn.cmo test_nn.cmo

train_loan: matrix.cmo nn.cmo train_loan.cmo
	$(OCAML) -o train_loan matrix.cmo nn.cmo train_loan.cmo

# ---- Clean ----

clean:
	rm -f *.cmo *.cmi test_matrix test_nn train_loan a.out

# ---- Rebuild ----

re: clean all

