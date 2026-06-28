import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a logistic regression probe on CLIP features.")
    parser.add_argument("--features", required=True, help="Path to .npz file with features/labels/paths arrays")
    parser.add_argument("--output",   required=True, help="Output .pkl path for the fitted probe")
    parser.add_argument("--C",            type=float, default=1.0)
    parser.add_argument("--max_iter",     type=int,   default=1000)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--class_weight", default=None,
                        help="Set to 'balanced' to compensate for class imbalance")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.features, allow_pickle=True)
    X, y = data["features"], data["labels"]
    print(f"Loaded {len(X)} samples of shape {X.shape[1]}-d  "
          f"({(y == 0).sum()} real, {(y == 1).sum()} fake)")

    clf = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=args.seed,
                             class_weight=args.class_weight)
    clf.fit(X, y)

    train_acc = clf.score(X, y)
    print(f"Train accuracy: {train_acc:.4f}  (n_iter={clf.n_iter_[0]})")

    joblib.dump(clf, output_path)
    print(f"Probe saved → {output_path}")
