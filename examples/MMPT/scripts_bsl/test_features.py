import argparse
import time
from tqdm import tqdm
import numpy as np
import tensorflow_datasets as tfds
from sign_language_datasets.datasets.config import SignDatasetConfig
from sklearn.linear_model import LogisticRegression
import glob
import os
import re
import lmdb
from lmdb_loader import LMDBLoader

def load_embedding_lookup(directory):
    """
    Load all numpy files matching video_embeddings_*.npy in the given directory,
    then concatenate them along the first axis to form a lookup table.
    """
    pattern = f"{directory}/video_embeddings_*.npy"
    files = glob.glob(pattern)
    # Sort files by the integer part in the filename.
    files = sorted(files, key=lambda f: int(re.search(r'video_embeddings_(\d+)\.npy', f).group(1)))
    arrays = [np.load(f) for f in files]
    return np.concatenate(arrays, axis=0)

def get_sign_clip_embedding(ex, idx, sign_clip_from_episode, embeddings, pooling_method):
    """
    Retrieve the sign_clip embedding for an example.
    If sign_clip_from_episode is True, load the corresponding episode file,
    adapt the start and end frames using a fixed stride of 4, and pool the features
    using the chosen pooling method. Otherwise, retrieve the precomputed embedding from
    the lookup table.
    If the episode file is missing, return None.
    """
    if sign_clip_from_episode:
        episode_id = ex['episode_id'].numpy().decode('utf-8')
        start_frame = int(ex['start_frame'])
        end_frame = int(ex['end_frame'])
        sign_clip_file = f"/scratch/shared/beegfs/zifan/bobsl/video_features/sign_clip_bobsl/{episode_id}.npy"
        if not os.path.exists(sign_clip_file):
            print(f"Warning: sign_clip file {sign_clip_file} not found. Skipping example {idx}.")
            return None
        stride = 4
        adjusted_start = start_frame // stride
        adjusted_end = end_frame // stride
        emb_array = np.load(sign_clip_file)[adjusted_start:adjusted_end]
        if pooling_method == "max":
            pooled = np.max(emb_array, axis=0).astype(np.float32)
        else:
            pooled = np.mean(emb_array, axis=0).astype(np.float32)
        return pooled
    else:
        return embeddings[idx].astype(np.float32)

def get_swin_embedding(ex, idx, pooling_method, lmdb_loader):
    """
    Retrieve the swin embedding for an example from an LMDB database.
    Uses the provided LMDBLoader to load the feature sequence for the given episode,
    from begin_frame to end_frame. Pools the resulting features using the specified
    pooling method.
    The loaded features (a torch tensor) are converted to a numpy array before pooling.
    """
    episode_id = ex['episode_id'].numpy().decode('utf-8')
    start_frame = int(ex['start_frame'])
    end_frame = int(ex['end_frame'])
    try:
        episode_features = lmdb_loader.load_sequence(
            episode_name=episode_id,
            begin_frame=start_frame,
            end_frame=end_frame,
        )
        # Convert torch tensor to numpy array.
        if hasattr(episode_features, "detach"):
            episode_features = episode_features.detach().cpu().numpy()
        else:
            episode_features = episode_features.numpy()
    except Exception as e:
        print(f"Warning: Failed to load swin features for episode {episode_id} at index {idx}: {e}")
        return None
    if pooling_method == "max":
        pooled = np.max(episode_features, axis=0).astype(np.float32)
    else:
        pooled = np.mean(episode_features, axis=0).astype(np.float32)
    return pooled

def process_examples(dataset, desc, debug, embeddings=None, features=["lip"], 
                     sign_clip_from_episode=False, pooling_method="avg", lmdb_loader=None):
    """
    Process each example in the dataset by computing and concatenating the requested feature embeddings.
    If any requested feature fails for an example, that example is skipped.
    """
    examples = []
    for idx, ex in enumerate(tqdm(dataset, desc=desc)):
        if debug and idx >= 10000:
            break
        feature_embeddings = []
        for feat in features:
            if feat == "lip":
                lip_data = ex['lip'].numpy()
                if pooling_method == "max":
                    lip_feature = np.max(lip_data, axis=0).astype(np.float32)
                else:
                    lip_feature = np.mean(lip_data, axis=0).astype(np.float32)
                feature_embeddings.append(lip_feature)
            elif feat == "sign_clip":
                sign_clip_emb = get_sign_clip_embedding(ex, idx, sign_clip_from_episode, embeddings, pooling_method)
                if sign_clip_emb is None:
                    feature_embeddings = []  # skip this example if missing feature
                    break
                feature_embeddings.append(sign_clip_emb)
            elif feat == "swin":
                if lmdb_loader is None:
                    print("Error: LMDBLoader is not initialized for swin features.")
                    feature_embeddings = []  # skip this example
                    break
                swin_emb = get_swin_embedding(ex, idx, pooling_method, lmdb_loader)
                if swin_emb is None:
                    feature_embeddings = []  # skip this example if missing feature
                    break
                feature_embeddings.append(swin_emb)
        if len(feature_embeddings) != len(features):
            continue  # skip example if any feature was not successfully loaded
        # Concatenate features if more than one type was requested.
        if len(feature_embeddings) > 1:
            concatenated = np.concatenate(feature_embeddings)
        else:
            concatenated = feature_embeddings[0]
        new_example = {
            'feature': concatenated,
            'text': ex['text'].numpy().decode('utf-8')
        }
        examples.append(new_example)
        # print(new_example['text'])
    return examples

def split_examples(examples, test_ratio=0.1, random_seed=42):
    """
    Randomly splits the examples list into training and testing sets.
    test_ratio determines the fraction of examples that go to the test set.
    """
    np.random.seed(random_seed)
    indices = np.random.permutation(len(examples))
    test_size = int(len(examples) * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    train_split = [examples[i] for i in train_indices]
    test_split = [examples[i] for i in test_indices]
    return train_split, test_split

if __name__ == "__main__":
    # -------------------- Argument Parsing --------------------
    parser = argparse.ArgumentParser(
        description="Feature Extraction and Logistic Regression Training (CPU Only)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If set, only process 100 examples for faster debugging."
    )
    parser.add_argument(
        "--dataset_for_training",
        type=str,
        choices=["training", "validation", "test"],
        default="training",
        help="Which dataset to use for training. Default is 'training'."
    )
    parser.add_argument(
        "--dataset_for_test",
        type=str,
        choices=["training", "validation", "test"],
        default="test",
        help="Which dataset to use for testing. Default is 'test'."
    )
    parser.add_argument(
        "--feature",
        type=str,
        nargs="+",
        choices=["lip", "sign_clip", "swin"],
        default=["lip"],
        help="Feature types to use. You can specify one or more from: 'lip', 'sign_clip', 'swin'. When more than one is provided, features are concatenated."
    )
    parser.add_argument(
        "--sign_clip_from_episode",
        action="store_true",
        help="If set, load sign_clip embeddings from episode files using adapted frame indices and pooling instead of using precomputed lookup tables."
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        choices=["avg", "max"],
        default="avg",
        help="Pooling method to use for features: 'avg' for average pooling (default) or 'max' for max pooling."
    )
    # New CLI argument for selecting the solver.
    parser.add_argument(
        "--solver",
        type=str,
        choices=["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        default="lbfgs",
        help="Solver to use for logistic regression. Default is 'lbfgs'."
    )
    # New arguments for swin LMDB features.
    parser.add_argument(
        "--swin_lmdb_path",
        type=str,
        default="/users/zifan/BOBSL/derivatives/video_features/swin_v2/lmdb-feats_vswin_t-bs256_float16",
        help="Path to the LMDB database for swin features."
    )
    parser.add_argument(
        "--swin_input_features_stride",
        type=int,
        default=4,
        help="Input features stride for swin features."
    )
    args = parser.parse_args()
    
    # Print all CLI options before proceeding.
    print("CLI options:", args)
    
    debug = args.debug
    features = args.feature  # This is now a list of feature types.
    pooling_method = args.pooling_method

    # -------------------- Data Loading Benchmark --------------------
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
    print("Loading dataset...")
    start_data_time = time.perf_counter()
    data = tfds.load(name='bobsl_islr', builder_kwargs=dict(config=sd_config), data_dir=data_dir)
    data_load_time = time.perf_counter() - start_data_time
    print(f"Time to load dataset: {data_load_time:.2f} seconds")
    
    # -------------------- Embedding Lookup Tables --------------------
    # For sign_clip features, only load precomputed embeddings if not using the episode-based loading.
    if ("sign_clip" in features) and (not args.sign_clip_from_episode):
        print("Loading precomputed embeddings...")
        emb_start_time = time.perf_counter()
        if args.dataset_for_training == "validation":
            train_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_valid"
        elif args.dataset_for_training == "test":
            train_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_test"
        else:  # "training"
            train_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_train"
        
        if args.dataset_for_test == "training":
            test_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_train"
        elif args.dataset_for_test == "validation":
            test_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_valid"
        elif args.dataset_for_test == "test":
            test_emb_dir = "/scratch/shared/beegfs/zifan/runs/retri_bsl/bobsl_islr_finetune/eval_v3/bobsl_islr_test"
            
        train_embeddings = load_embedding_lookup(train_emb_dir)
        test_embeddings = load_embedding_lookup(test_emb_dir)
        emb_load_time = time.perf_counter() - emb_start_time
        print(f"Time to load embeddings: {emb_load_time:.2f} seconds")
    else:
        train_embeddings = None
        test_embeddings = None

    # -------------------- Initialize LMDBLoader for swin features if needed --------------------
    lmdb_loader = None
    if "swin" in features:
        print("Initializing LMDB loader for swin features...")
        lmdb_stride = 2
        lmdb_loader = LMDBLoader(
            lmdb_path=args.swin_lmdb_path,
            load_type="feats",
            feat_dim=768,
            lmdb_stride=lmdb_stride,
            load_stride=int(args.swin_input_features_stride / lmdb_stride),
        )

    # -------------------- Prepare Training and Testing Examples --------------------
    print("Processing examples...")
    start_proc_time = time.perf_counter()
    if args.dataset_for_training == args.dataset_for_test:
        print(f"Using {args.dataset_for_training} split for both training and testing. Randomly splitting examples with a 90:10 ratio.")
        if args.dataset_for_training == "validation" and "validation" in data:
            examples = process_examples(
                data['validation'], "Loading examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_training == "test" and "test" in data:
            examples = process_examples(
                data['test'], "Loading examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_training == "training" and "train" in data:
            examples = process_examples(
                data['train'], "Loading examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        else:
            raise ValueError("Selected dataset split not found in data.")
        
        train_examples, test_examples = split_examples(examples, test_ratio=0.1, random_seed=42)
    else:
        if args.dataset_for_training == "validation" and "validation" in data:
            print("Using validation set for training.")
            train_examples = process_examples(
                data['validation'], "Loading training examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_training == "test" and "test" in data:
            print("Using test split for training.")
            train_examples = process_examples(
                data['test'], "Loading training examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_training == "training" and "train" in data:
            print("Using train split for training.")
            train_examples = process_examples(
                data['train'], "Loading training examples", debug,
                embeddings=train_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        else:
            raise ValueError("Selected training dataset split not found in data.")
        
        if args.dataset_for_test == "training" and "train" in data:
            print("Using train split for testing.")
            test_examples = process_examples(
                data['train'], "Loading testing examples", debug,
                embeddings=test_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_test == "validation" and "validation" in data:
            print("Using validation set for testing.")
            test_examples = process_examples(
                data['validation'], "Loading testing examples", debug,
                embeddings=test_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        elif args.dataset_for_test == "test" and "test" in data:
            print("Using test split for testing.")
            test_examples = process_examples(
                data['test'], "Loading testing examples", debug,
                embeddings=test_embeddings, features=features,
                sign_clip_from_episode=args.sign_clip_from_episode,
                pooling_method=pooling_method,
                lmdb_loader=lmdb_loader
            )
        else:
            raise ValueError("Selected testing dataset split not found in data.")
    proc_time = time.perf_counter() - start_proc_time
    print(f"Time to process examples: {proc_time:.2f} seconds")
    
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
    
    clf = LogisticRegression(verbose=True, random_state=42, max_iter=200, solver=args.solver)
    
    start_train_time = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train_time
    print(f"Time to train model: {train_time:.2f} seconds")
    
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
