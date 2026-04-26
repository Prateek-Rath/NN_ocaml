import time
import random
import os
from nn import Model, Dense, ReLU, Softmax

class LoanDataset:
    def __init__(self, filename, max_samples=500000):
        self.filename = filename
        self.max_samples = max_samples
        self.x = None
        self.y = None

    @staticmethod
    def map_gender(g):
        return 1.0 if g == "female" else 0.0

    @staticmethod
    def map_education(e):
        mapping = {"Master": 4.0, "Doctorate": 3.0, "Bachelor": 2.0, "Associate": 1.0, "High School": 0.0}
        return mapping.get(e, 0.0)

    @staticmethod
    def map_ownership(o):
        mapping = {"MORTGAGE": 3.0, "OWN": 2.0, "RENT": 1.0, "OTHER": 0.0}
        return mapping.get(o, 0.0)

    @staticmethod
    def map_intent(i):
        mapping = {"PERSONAL": 0.0, "EDUCATION": 1.0, "MEDICAL": 2.0, "VENTURE": 3.0, "HOMEIMPROVEMENT": 4.0, "DEBTCONSOLIDATION": 5.0}
        return mapping.get(i, 0.0)

    @staticmethod
    def map_defaults(d):
        return 1.0 if d == "Yes" else 0.0

    @staticmethod
    def parse_float(s):
        try:
            return float(s)
        except ValueError:
            return 0.0

    def parse_features(self, cols):
        age, gender, ed, inc, emp, own, amnt, intent, int_r, perc_inc, cred_l, cred_s, def_, status = cols
        return [
            self.parse_float(age), self.map_gender(gender), self.map_education(ed), self.parse_float(inc),
            self.parse_float(emp), self.map_ownership(own), self.parse_float(amnt), self.map_intent(intent),
            self.parse_float(int_r), self.parse_float(perc_inc), self.parse_float(cred_l), self.parse_float(cred_s),
            self.map_defaults(def_)
        ]

    def load_and_scale(self):
        print("Loading dataset (full)...", flush=True)
        x_list = []
        y_list = []
        count = 0
        header_skipped = False
        
        with open(self.filename, 'r') as f:
            for line in f:
                if count >= self.max_samples:
                    break
                line = line.strip()
                if not header_skipped:
                    header_skipped = True
                    continue
                if line == "":
                    continue
                cols = line.split(',')
                if len(cols) == 14:
                    features = self.parse_features(cols)
                    label = [0.0, 1.0] if cols[-1] == "1" else [1.0, 0.0]
                    x_list.append(features)
                    y_list.append(label)
                    count += 1
        print(f"Loaded {count} samples.", flush=True)
        
        if not x_list: return x_list, y_list
        cols = len(x_list[0])
        min_vals = [min(row[i] for row in x_list) for i in range(cols)]
        max_vals = [max(row[i] for row in x_list) for i in range(cols)]
        diffs = [(mx - mn) if (mx - mn) != 0.0 else 1.0 for mx, mn in zip(max_vals, min_vals)]
        
        self.x = [[(val - mn) / diff for val, mn, diff in zip(row, min_vals, diffs)] for row in x_list]
        self.y = y_list
        return self.x, self.y

    def shuffle(self):
        combined = list(zip(self.x, self.y))
        random.shuffle(combined)
        shuffled_x, shuffled_y = zip(*combined)
        return list(shuffled_x), list(shuffled_y)

class ModelEvaluator:
    @staticmethod
    def evaluate(model, x, y):
        preds = model.forward(x)
        final_loss = model.cross_entropy(preds, y)
        print(f"Final model loss: {final_loss:.6f}")
        
        correct = 0
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        
        for p, t in zip(preds, y):
            pred_class = 1 if p[1] > p[0] else 0
            target_class = 1 if t[1] == 1.0 else 0
            
            if pred_class == target_class: correct += 1
            if pred_class == 1 and target_class == 1: tp += 1
            if pred_class == 1 and target_class == 0: fp += 1
            if pred_class == 0 and target_class == 0: tn += 1
            if pred_class == 0 and target_class == 1: fn += 1
            
        total = len(y)
        accuracy = (correct / total) * 100.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

def main():
    random.seed()
    csv_path = os.path.join("..", "ocaml_implementation", "loan_data.csv")
    dataset = LoanDataset(csv_path, max_samples=500000)
    x, y = dataset.load_and_scale()
    shuffled_x, shuffled_y = dataset.shuffle()
    
    model = Model()
    model.add(Dense(13, 16))
    model.add(ReLU())
    model.add(Dense(16, 2))
    model.add(Softmax())
    
    print("Starting training...", flush=True)
    start_time = time.time()
    epochs = 20
    batch_size = 32
    lr = 0.01
    
    model.train(shuffled_x, shuffled_y, batch_size, epochs, lr)
    
    end_time = time.time()
    ModelEvaluator.evaluate(model, x, y)
    
    train_time = end_time - start_time
    print(f"Time taken: {train_time:.2f} seconds")
    print(f"Throughput: {(len(y) * epochs) / train_time:.2f} samples/sec")

if __name__ == "__main__":
    main()
