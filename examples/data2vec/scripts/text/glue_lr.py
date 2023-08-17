import os.path as osp
import re
from collections import defaultdict

from valids import parser, main as valids_main


TASK_TO_METRIC = {
    "cola": "mcc",
    "qnli": "accuracy",
    "mrpc": "acc_and_f1",
    "rte": "accuracy",
    "sst_2": "accuracy",
    "mnli": "accuracy",
    "qqp": "acc_and_f1",
    "sts_b": "pearson_and_spearman",
}
TASKS = ["cola", "qnli", "mrpc", "rte", "sst_2", "mnli", "qqp", "sts_b"]


def get_best_stat_str(task_vals, show_subdir):
    task_to_best_val = {}
    task_to_best_dir = {}
    for task, subdir_to_val in task_vals.items():
        task_to_best_val[task] = max(subdir_to_val.values())
        task_to_best_dir[task] = max(subdir_to_val.keys(), key=lambda x: subdir_to_val[x])

    # import pdb; pdb.set_trace()
    N1 = len(task_to_best_val)
    N2 = len([k for k in task_to_best_val if k != "rte"])
    avg1 = sum(task_to_best_val.values()) / N1
    avg2 = sum(v for task, v in task_to_best_val.items() if task != "rte") / N2

    try:
        msg = ""
        for task in TASKS:
            dir = task_to_best_dir.get(task, 'null')
            val = task_to_best_val.get(task, -100)
            msg += f"({dir}, {val})\t" if show_subdir else f"{val}\t"
        msg += f"{avg1:.2f}\t{avg2:.2f}"
    except Exception as e:
        msg = str(e)
        msg += str(sorted(task_vals.items()))
    return msg

def get_all_stat_str(task_vals):
    msg = ""
    for task in [task for task in TASKS if task in task_vals]:
        msg += f"=== {task}\n"
        for subdir in sorted(task_vals[task].keys()):
            msg += f"\t{subdir}\t{task_vals[task][subdir]}\n"
    return msg

def get_tabular_stat_str(task_vals):
    """assume subdir is <param>/run_*/0"""
    msg = ""
    for task in [task for task in TASKS if task in task_vals]:
        msg += f"=== {task}\n"
        param_to_runs = defaultdict(dict)
        for subdir in task_vals[task]:
            match = re.match("(.*)/(run_.*)/0", subdir)
            assert match, "subdir"
            param, run = match.groups()
            param_to_runs[param][run] = task_vals[task][subdir]
        params = sorted(param_to_runs, key=lambda x: float(x))
        runs = sorted(set(run for runs in param_to_runs.values() for run in runs))
        msg += ("runs:" + "\t".join(runs) + "\n")
        msg += ("params:" + "\t".join(params) + "\n")
        for param in params:
            msg += "\t".join([str(param_to_runs[param].get(run, None)) for run in runs])
            msg += "\n"
        # for subdir in sorted(task_vals[task].keys()):
        #     msg += f"\t{subdir}\t{task_vals[task][subdir]}\n"
    return msg

   

def main():
    parser.add_argument("--show_glue", action="store_true", help="show glue metric for each task instead of accuracy")
    parser.add_argument("--print_mode", default="best", help="best|all|tabular")
    parser.add_argument("--show_subdir", action="store_true", help="print the subdir that has the best results for each run")
    parser.add_argument("--override_target", default="valid_accuracy", help="override target")

    args = parser.parse_args()
    args.target = args.override_target
    args.best_biggest = True
    args.best = True
    args.last = 0
    args.path_contains = None
    
    res =  valids_main(args, print_output=False)
    grouped_acc = {}
    grouped_met = {}  # use official metric for each task
    for path, v in res.items():
        path = "/".join([args.base, path])
        path = re.sub("//*", "/", path)
        match = re.match("(.*)finetune[^/]*/([^/]*)/(.*)", path)
        if not match:
            continue
        run, task, subdir = match.groups()

        if run not in grouped_acc:
            grouped_acc[run] = {}
            grouped_met[run] = {}
        if task not in grouped_acc[run]:
            grouped_acc[run][task] = {}
            grouped_met[run][task] = {}

        if v is not None:
            grouped_acc[run][task][subdir] = float(v.get("valid_accuracy", -100))
            grouped_met[run][task][subdir] = float(v.get(f"valid_{TASK_TO_METRIC[task]}", -100))
        else:
            print(f"{path} has None return")

    header = "\t".join(TASKS)
    for run in sorted(grouped_acc):
        print(run)
        if args.print_mode == "all":
            if args.show_glue:
                print("===== GLUE =====")
                print(get_all_stat_str(grouped_met[run]))
            else:
                print("===== ACC =====")
                print(get_all_stat_str(grouped_acc[run]))
        elif args.print_mode == "best":
            print(f"      {header}")
            if args.show_glue:
                print(f"GLEU: {get_best_stat_str(grouped_met[run], args.show_subdir)}")
            else:
                print(f"ACC:  {get_best_stat_str(grouped_acc[run], args.show_subdir)}")
        elif args.print_mode == "tabular":
            if args.show_glue:
                print("===== GLUE =====")
                print(get_tabular_stat_str(grouped_met[run]))
            else:
                print("===== ACC =====")
                print(get_tabular_stat_str(grouped_acc[run]))
        else:
            raise ValueError(args.print_mode)
        print()

if __name__ == "__main__":
    main()
