import numpy as np
import time
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
        x_list, y_list = [], []
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
        
        raw_x = np.array(x_list, dtype=np.float64)
        self.y = np.array(y_list, dtype=np.float64)
        
        # Scale X using Min-Max scaling
        min_vals = np.min(raw_x, axis=0, keepdims=True)
        max_vals = np.max(raw_x, axis=0, keepdims=True)
        diffs = max_vals - min_vals
        diffs[diffs == 0.0] = 1.0
        self.x = (raw_x - min_vals) / diffs
        
        return self.x, self.y

    def shuffle(self):
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        return self.x[indices], self.y[indices]


class ModelEvaluator:
    @staticmethod
    def evaluate(model, x, y):
        # Forward pass
        preds = model.forward(x)
        final_loss = model.cross_entropy(preds, y)
        print(f"Final model loss: {final_loss:.6f}")
        
        # Metrics
        pred_class = (preds[:, 1] > preds[:, 0]).astype(int)
        target_class = (y[:, 1] == 1.0).astype(int)
        
        correct = np.sum(pred_class == target_class)
        tp = np.sum((pred_class == 1) & (target_class == 1))
        fp = np.sum((pred_class == 1) & (target_class == 0))
        tn = np.sum((pred_class == 0) & (target_class == 0))
        fn = np.sum((pred_class == 0) & (target_class == 1))
        
        total = len(y)
        accuracy = (correct / total) * 100.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) > 0.0 else 0.0
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")


def main():
    np.random.seed()
    
    # Dataset Preparation
    csv_path = os.path.join("..", "ocaml_implementation", "loan_data.csv")
    dataset = LoanDataset(csv_path, max_samples=500000)
    x, y = dataset.load_and_scale()
    shuffled_x, shuffled_y = dataset.shuffle()
    
    # Model Configuration
    model = Model()
    model.add(Dense(13, 16))
    model.add(ReLU())
    model.add(Dense(16, 2))
    model.add(Softmax())
    
    # Training
    print("Starting training...", flush=True)
    start_time = time.time()
    epochs = 20
    batch_size = 32
    lr = 0.01
    
    model.train(shuffled_x, shuffled_y, batch_size, epochs, lr)
    
    end_time = time.time()
    
    # Evaluation
    ModelEvaluator.evaluate(model, x, y)
    
    train_time = end_time - start_time
    print(f"Time taken: {train_time:.2f} seconds")
    print(f"Throughput: {(len(y) * epochs) / train_time:.2f} samples/sec")


if __name__ == "__main__":
    main()
