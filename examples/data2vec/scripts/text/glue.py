from valids import parser, main as valids_main
import os.path as osp


args = parser.parse_args()
args.target = "valid_accuracy"
args.best_biggest = True
args.best = True
args.last = 0
args.path_contains = None

res =  valids_main(args, print_output=False)

grouped = {}
for k, v in res.items():
    k = osp.dirname(k)
    run = osp.dirname(k)
    task = osp.basename(k)
    val = v["valid_accuracy"]

    if run not in grouped:
        grouped[run] = {}

    grouped[run][task] = val

for run, tasks in grouped.items():
    print(run)
    avg = sum(float(v) for v in tasks.values()) / len(tasks)
    avg_norte = sum(float(v) for k,v in tasks.items() if k != 'rte') / (len(tasks) -1)
    try:
        print(f"{tasks['cola']}\t{tasks['qnli']}\t{tasks['mrpc']}\t{tasks['rte']}\t{tasks['sst_2']}\t{avg:.2f}\t{avg_norte:.2f}")
    except:
        print(tasks)
    print()
