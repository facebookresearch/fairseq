import os, argparse, re, json, copy, math
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('base', help='base log path')
parser.add_argument('--file_name', default='train.log', help='the log file name')
parser.add_argument('--target', default='valid_loss', help='target metric')
parser.add_argument('--last', type=int, default=999999999, help='print last n matches')
parser.add_argument('--last_files', type=int, default=None, help='print last x files')
parser.add_argument('--everything', action='store_true', help='print everything instead of only last match')
parser.add_argument('--path_contains', help='only consider matching file pattern')
parser.add_argument('--group_on', help='if set, groups by this metric and shows table of differences')
parser.add_argument('--epoch', help='epoch for comparison', type=int)
parser.add_argument('--skip_empty', action='store_true', help='skip empty results')
parser.add_argument('--skip_containing', help='skips entries containing this attribute')
parser.add_argument('--unique_epochs', action='store_true', help='only consider the last line fore each epoch')
parser.add_argument('--best', action='store_true', help='print the last best result')
parser.add_argument('--avg_params', help='average these params through entire log')
parser.add_argument('--extract_prev', help='extracts this metric from previous line')

parser.add_argument('--remove_metric', help='extracts this metric from previous line')

parser.add_argument('--compact', action='store_true', help='if true, just prints checkpoint <tab> best val')
parser.add_argument('--hydra', action='store_true', help='if true, uses hydra param conventions')

parser.add_argument('--best_biggest', action='store_true', help='if true, best is the biggest number, not smallest')
parser.add_argument('--key_len', type=int, default=10, help='max length of key')

parser.add_argument('--best_only', action='store_true', help='if set, only prints the best value')
parser.add_argument('--flat', action='store_true', help='just print the best results')


def main(args, print_output):
    ret = {}

    entries = []

    def extract_metric(s, metric):
        try:
            j = json.loads(s)
        except:
            return None
        if args.epoch is not None and ('epoch' not in j or j['epoch'] != args.epoch):
            return None
        return j[metric] if metric in j else None


    def extract_params(s):
        s = s.replace(args.base, '', 1)
        if args.path_contains is not None:
            s = s.replace(args.path_contains, '', 1)

        if args.hydra:
            num_matches = re.findall(r'(?:/|__)([^/:]+):(\d+\.?\d*)', s)
            # str_matches = re.findall(r'(?:/|__)([^/:]+):([^\.]*[^\d\.]+)(?:/|__)', s)
            str_matches = re.findall(r'(?:/|__)?((?:(?!(?:\:|__)).)+):([^\.]*[^\d\.]+\d*)(?:/|__)', s)
            lr_matches =  re.findall(r'optimization.(lr):\[([\d\.,]+)\]', s)
            task_matches = re.findall(r'.*/(\d+)$', s)
        else:
            num_matches = re.findall(r'\.?([^\.]+?)(\d+(e\-\d+)?(?:\.\d+)?)(\.|$)', s)
            str_matches = re.findall(r'[/\.]([^\.]*[^\d\.]+\d*)(?=\.)', s)
            lr_matches = []
            task_matches = []

        cp_matches = re.findall(r'checkpoint(?:_\d+)?_(\d+).pt', s)

        items = OrderedDict()
        for m in str_matches:
            if isinstance(m, tuple):
                if 'checkpoint' not in m[0]:
                    items[m[0]] = m[1]
            else:
                items[m] = ''

        for m in num_matches:
            items[m[0]] = m[1]

        for m in lr_matches:
            items[m[0]] = m[1]

        for m in task_matches:
            items["hydra_task"] = m

        for m in cp_matches:
            items['checkpoint'] = m

        return items

    abs_best = None

    sources = []
    for root, _, files in os.walk(args.base):
        if args.path_contains is not None and not args.path_contains in root:
            continue
        for f in files:
            if f.endswith(args.file_name):
                sources.append((root, f))

    if args.last_files is not None:
        sources = sources[-args.last_files:]

    for root, file in sources:
        with open(os.path.join(root, file), 'r') as fin:
            found = []
            avg = {}
            prev = None
            for line in fin:
                line = line.rstrip()
                if line.find(args.target) != -1 and (
                        args.skip_containing is None or line.find(args.skip_containing) == -1):
                    try:
                        idx = line.index("{")
                        line = line[idx:]
                        line_json = json.loads(line)
                    except:
                        continue
                    if prev is not None:
                        try:
                            prev.update(line_json)
                            line_json = prev
                        except:
                            pass
                    if args.target in line_json:
                        found.append(line_json)
                if args.avg_params:
                    avg_params = args.avg_params.split(',')
                    for p in avg_params:
                        m = extract_metric(line, p)
                        if m is not None:
                            prev_v, prev_c = avg.get(p, (0, 0))
                            avg[p] = prev_v + float(m), prev_c + 1
                if args.extract_prev:
                    try:
                        prev = json.loads(line)
                    except:
                        pass
            best = None
            if args.best:
                curr_best = None
                for i in range(len(found)):
                    cand_best = found[i][args.target] if args.target in found[i] else None

                    def cmp(a, b):
                        a = float(a)
                        b = float(b)
                        if args.best_biggest:
                            return a > b
                        return a < b

                    if cand_best is not None and not math.isnan(float(cand_best)) and (
                            curr_best is None or cmp(cand_best, curr_best)):
                        curr_best = cand_best
                        if abs_best is None or cmp(curr_best, abs_best):
                            abs_best = curr_best
                        best = found[i]
            if args.unique_epochs or args.epoch:
                last_found = []
                last_epoch = None
                for i in reversed(range(len(found))):
                    epoch = found[i]['epoch']
                    if args.epoch and args.epoch != epoch:
                        continue
                    if epoch != last_epoch:
                        last_epoch = epoch
                        last_found.append(found[i])
                found = list(reversed(last_found))

            if len(found) == 0:
                if print_output and (args.last_files is not None or not args.skip_empty):
                    # print(root.split('/')[-1])
                    print(root[len(args.base):])
                    print('Nothing')
            else:
                if not print_output:
                    ret[root[len(args.base):]] = best
                    continue

                if args.compact:
                    # print('{}\t{}'.format(root.split('/')[-1], curr_best))
                    print('{}\t{}'.format(root[len(args.base)+1:], curr_best))
                    continue

                if args.group_on is None and not args.best_only:
                    # print(root.split('/')[-1])
                    print(root[len(args.base):])
                if not args.everything:
                    if best is not None and args.group_on is None and not args.best_only and not args.flat:
                        print(best, '(best)')
                    if args.group_on is None and args.last and not args.best_only and not args.flat:
                        for f in found[-args.last:]:
                            if args.extract_prev is not None:
                                try:
                                    print('{}\t{}'.format(f[args.extract_prev], f[args.target]))
                                except Exception as e:
                                    print('Exception!', e)
                            else:
                                print(f)
                    try:
                        metric = found[-1][args.target] if not args.best or best is None else best[args.target]
                    except:
                        print(found[-1])
                        raise
                    if metric is not None:
                        entries.append((extract_params(root), metric))
                else:
                    for f in found:
                        print(f)
                if not args.group_on and print_output:
                    print()

            if len(avg) > 0:
                for k, (v, c) in avg.items():
                    print(f'{k}: {v/c}')

    if args.best_only:
        print(abs_best)

    if args.flat:
        print("\t".join(m for _, m in entries))

    if args.group_on is not None:
        by_val = OrderedDict()
        for e, m in entries:
            k = args.group_on
            if k not in e:
                m_keys = [x for x in e.keys() if x.startswith(k)]
                if len(m_keys) == 0:
                    val = "False"
                else:
                    assert len(m_keys) == 1
                    k = m_keys[0]
                    val = m_keys[0]
            else:
                val = e[args.group_on]
                if val == "":
                    val = "True"
            scrubbed_entry = copy.deepcopy(e)
            if k in scrubbed_entry:
                del scrubbed_entry[k]
            if args.remove_metric and args.remove_metric in scrubbed_entry:
                val += '_' + scrubbed_entry[args.remove_metric]
                del scrubbed_entry[args.remove_metric]
            by_val.setdefault(tuple(scrubbed_entry.items()), dict())[val] = m
        distinct_vals = set()
        for v in by_val.values():
            distinct_vals.update(v.keys())
        try:
            distinct_vals = {int(d) for d in distinct_vals}
        except:
            print(distinct_vals)
            print()
            print("by_val", len(by_val))
            for k,v in by_val.items():
                print(k, '=>', v)
            print()

            # , by_val, entries)
            raise
        from natsort import natsorted
        svals = list(map(str, natsorted(distinct_vals)))
        print('{}\t{}'.format(args.group_on, '\t'.join(svals)))
        sums = OrderedDict({n:[] for n in svals})
        for k, v in by_val.items():
            kstr = '.'.join(':'.join(x) for x in k)
            vstr = ''
            for mv in svals:
                x = v[mv] if mv in v else ''
                vstr += '\t{}'.format(round(x, 5) if isinstance(x, float) else x)
                try:
                    sums[mv].append(float(x))
                except:
                    pass
            print('{}{}'.format(kstr[:args.key_len], vstr))
        if any(len(x) > 0 for x in sums.values()):
            print('min:', end='')
            for v in sums.values():
                min = np.min(v)
                print(f'\t{round(min, 5)}', end='')
            print()
            print('max:', end='')
            for v in sums.values():
                max = np.max(v)
                print(f'\t{round(max, 5)}', end='')
            print()
            print('avg:', end='')
            for v in sums.values():
                mean = np.mean(v)
                print(f'\t{round(mean, 5)}', end='')
            print()
            print('median:', end='')
            for v in sums.values():
                median = np.median(v)
                print(f'\t{round(median, 5)}', end='')
            print()

    return ret

if __name__ == "__main__":
    args = parser.parse_args()
    main(args, print_output=True)