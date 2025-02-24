import argparse
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
from sklearn.linear_model import LogisticRegression
import glob

def load_embedding_lookup(directory):
    """
    Load all numpy files matching video_embeddings_*.npy in the given directory,
    then concatenate them along the first axis to form a lookup table.
    """
    pattern = f"{directory}/video_embeddings_*.npy"
    files = glob.glob(pattern)
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)

def process_examples(dataset, desc, debug, embeddings=None, feature="lip"):
    examples = []
    for idx, ex in enumerate(tqdm(dataset, desc=desc)):
        if debug and idx >= 100:
            break
        if feature == "lip":
            # Compute the average along the first axis and assign to 'feature'
            ex['feature'] = ex['lip'].numpy().mean(axis=0)
        elif feature == "sign_clip":
            # Use the embedding lookup table: assign the corresponding row as the feature.
            ex['feature'] = embeddings[idx]
        elif feature == "both":
            # Compute lip feature and get the sign_clip feature, then concatenate.
            lip_feature = ex['lip'].numpy().mean(axis=0)
            sign_clip_feature = embeddings[idx]
            ex['feature'] = np.concatenate([lip_feature, sign_clip_feature])
        ex['text'] = ex['text'].numpy().decode('utf-8')
        examples.append(ex)
    return examples

if __name__ == "__main__":
    # -------------------- Argument Parsing --------------------
    parser = argparse.ArgumentParser(
        description="Lip Feature Extraction and Logistic Regression Training (CPU Only)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only process 100 examples for faster debugging."
    )
    parser.add_argument(
        "--use_validation_for_training",
        action="store_true",
        help="If set, use the validation set for training instead of the train split."
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["lip", "sign_clip", "both"],
        default="lip",
        help="Feature type to use. Default is 'lip'. If set to 'sign_clip', the embedding lookup will be used. If set to 'both', the lip and sign_clip features are concatenated."
    )
    args = parser.parse_args()
    debug = args.debug
    feature = args.feature

    data_dir = '/scratch/shared/beegfs/zifan/tensorflow_datasets'
    sd_config = SignDatasetConfig(
        name="holistic_lip", 
        include_video=False, 
        include_pose="holistic", 
        extra={
            'poses_dir': '/scratch/shared/beegfs/zifan/bobsl/video_features/mediapipe',
            'lip_feature_dir': '/scratch/shared/beegfs/zifan/bobsl/video_features/auto_asvr',
        },
    )

    data = tfds.load(name='bobsl_islr', builder_kwargs=dict(config=sd_config), data_dir=data_dir)
    
    # If using sign_clip or both, load the corresponding embedding lookup tables.
    if feature in ["sign_clip", "both"]:
        if args.use_validation_for_training:
            train_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_valid"
        else:
            train_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_train"
        test_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_test"
        train_embeddings = load_embedding_lookup(train_emb_dir)
        test_embeddings = load_embedding_lookup(test_emb_dir)
    else:
        train_embeddings = None
        test_embeddings = None

    # Choose training set based on flag; if not available, fallback to validation.
    if args.use_validation_for_training:
        print("Using validation set for training.")
        train_examples = process_examples(
            data['validation'], "Loading training examples", debug,
            embeddings=train_embeddings, feature=feature
        )
    else:
        if 'train' in data:
            print("Using train split for training.")
            train_examples = process_examples(
                data['train'], "Loading training examples", debug,
                embeddings=train_embeddings, feature=feature
            )
        else:
            print("Train split not found; using validation set for training.")
            train_examples = process_examples(
                data['validation'], "Loading training examples", debug,
                embeddings=train_embeddings, feature=feature
            )

    # Always use the test split for testing if available.
    if 'test' in data:
        test_examples = process_examples(
            data['test'], "Loading testing examples", debug,
            embeddings=test_embeddings, feature=feature
        )
    else:
        print("Test split not found; using validation set for testing.")
        test_examples = process_examples(
            data['validation'], "Loading testing examples", debug,
            embeddings=test_embeddings, feature=feature
        )

    # -------------------- Data Preparation --------------------
    X_train = np.array([ex['feature'] for ex in train_examples])
    X_test = np.array([ex['feature'] for ex in test_examples])
    y_train = np.array([ex['text'] for ex in train_examples])
    y_test = np.array([ex['text'] for ex in test_examples])
    
    print("Training set shape:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Testing set shape:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # -------------------- Filter for Common Labels --------------------
    common_labels = np.intersect1d(np.unique(y_train), np.unique(y_test))
    if len(common_labels) > 100:
        print("Common labels (showing first 100):", common_labels[:100])
    else:
        print("Common labels:", common_labels)
    
    train_mask = np.isin(y_train, common_labels)
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    
    test_mask = np.isin(y_test, common_labels)
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    print("After filtering for common labels:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # -------------------- CPU Training with scikit-learn --------------------
    print("Training using sklearn LogisticRegression on CPU.")
    num_classes = len(np.unique(y_train))
    print("Number of classes:", num_classes)
    
    clf = LogisticRegression(verbose=True, random_state=42, max_iter=200)
    clf.fit(X_train, y_train)
    top1_acc = clf.score(X_test, y_test)
    print(f"Top-1 accuracy (sklearn CPU): {top1_acc:.4f}")
    
    probs = clf.predict_proba(X_test)
    # Map the test labels to the corresponding indices of clf.classes_
    y_test_enc = np.array([np.where(clf.classes_ == label)[0][0] for label in y_test])
    
    def compute_topk_accuracy(probs, true_labels, k):
        topk_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = [true_label in topk for true_label, topk in zip(true_labels, topk_preds)]
        return np.mean(correct)
    
    top5 = compute_topk_accuracy(probs, y_test_enc, 5)
    top10 = compute_topk_accuracy(probs, y_test_enc, 10)
    print(f"Top-5 accuracy (sklearn CPU): {top5:.4f}")
    print(f"Top-10 accuracy (sklearn CPU): {top10:.4f}")
