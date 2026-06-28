import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit an MLP probe on CLIP features.")
    parser.add_argument("--features",    required=True, help="Path to .npz file with features/labels/paths arrays")
    parser.add_argument("--output",      required=True, help="Output .pkl path for the fitted probe")
    parser.add_argument("--hidden_sizes", default="512,128", help="Comma-separated hidden layer sizes")
    parser.add_argument("--max_iter",    type=int, default=500)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    X, y = data["features"], data["labels"]
    print(f"Loaded {len(X)} samples of shape {X.shape[1]}-d  "
          f"({(y == 0).sum()} real, {(y == 1).sum()} fake)")

    hidden_sizes = tuple(int(h) for h in args.hidden_sizes.split(","))
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_sizes,
        max_iter=args.max_iter,
        random_state=args.seed,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=True,
    )
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"Train accuracy: {train_acc:.4f}  (n_iter={clf.n_iter_})")

    joblib.dump(clf, output_path)
    print(f"Probe saved → {output_path}")
