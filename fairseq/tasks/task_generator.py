import numpy as np
from scipy import stats


class TaskGenerator():

    def __init__(self, max_tasks, max_samples, seqlen, vocab_size, num_classes, logdir=None):

        primes_below_1000 = []
        for i in range(2, 1000):
            isprime = True
            for prime in primes_below_1000:
                if i % prime == 0:
                    isprime = False
            if isprime:
                primes_below_1000.append(i)

        num_divisors = []
        for i in range(1, 20):
            num_div = 0
            for j in range(1, i + 1):
                if i % j == 0:
                    num_div += 1
            num_divisors.append(num_div)

        self.primes_below_1000 = primes_below_1000
        self.num_divisors = num_divisors

        self.construct_tasks()
        self.fn_map = self.function_mapping()

        self.vocab_size = vocab_size
        self.seqlen = seqlen
        self.max_tasks = max_tasks
        self.max_samples = max_samples

        self.num_classes = num_classes
        self.per_class_data = max_samples // self.num_classes
        self.thresh = self.num_classes * max_samples
        self.logdir = logdir

    def generate_tasks(self):

        tasks_explored = set()

        rng = np.random.RandomState(seed=1234)

        task_list = []
        task_descriptions = []

        tasks_iter = 0
        while True:
            print(tasks_iter)
            transform = self.transforms[rng.choice(len(self.transforms))]
            subseq = self.subseq[rng.choice(len(self.subseq))]
            labeling = self.labeling[rng.choice(len(self.labeling))]
            task_description = self.task_description(transform, subseq, labeling)
            data_strings = set()
            if task_description in tasks_explored:
                continue
            tasks_explored.add(task_description)
            data = {}
            labels_enough_data = set()
            num_iter = 0
            while True:
                num_iter += 1
                if num_iter > self.thresh:
                    break

                x = list(rng.randint(self.vocab_size, size=(self.seqlen,)))

                x_str = ' '.join(map(str, x))
                if x_str in data_strings:
                    continue
                data_strings.add(x_str)

                x_tf = self.fn_map[transform[0]](x, transform[1])

                flag = True
                if 'not' in subseq[0]:
                    flag = False
                x_subseq = self.fn_map[subseq[0].split()[-1]](x_tf, subseq[1], flag)
                if not x_subseq:  # Empty sequence
                    continue

                x_label = int(self.fn_map[labeling](x_subseq))
                if x_label in labels_enough_data:
                    continue

                if x_label not in data:
                    data[x_label] = []
                data[x_label].append(x)

                if len(data[x_label]) > self.per_class_data:
                    labels_enough_data.add(x_label)
                    if len(labels_enough_data) >= self.num_classes:
                        break

            if len(labels_enough_data) >= self.num_classes:
                data_list = []
                label_map = dict(zip(list(labels_enough_data), list(range(self.num_classes))))
                for label in labels_enough_data:
                    for instance in data[label]:
                        data_list.append((instance, label_map[label]))
                rng.shuffle(data_list)
                tasks_iter += 1

                task_list.append(data_list)
                task_descriptions.append(task_description)

            if tasks_iter >= self.max_tasks:
                break

        if self.logdir is not None:
            with open(self.logdir, 'w') as f:
                f.write('\n'.join(task_descriptions))

        return task_list

    def load_tasks(self, tasks_file):
        with open(tasks_file, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        return lines

    def generate_data(self, tasks, num_train, num_test, uniform_classes=False):

        rng = np.random.RandomState(seed=1234)

        all_data = []

        count = 0
        
        num_examples = num_train + 2 * num_test

        per_class_data = num_examples // self.num_classes

        for task in tasks:
            count += 1
            transform, subseq, labeling = task.split(' -> ')
            transform = transform.split()
            transform[1] = int(transform[1])

            subseq = subseq.split()
            if len(subseq) == 3:
                subseq = [' '.join(subseq[:2]), int(subseq[-1])]
            else:
                subseq[1] = int(subseq[1])

            data_strings = set()
            data = {}
            labels_enough_data = set()
            num_iter = 0

            while True:

                x = list(rng.randint(self.vocab_size, size=(self.seqlen,)))

                x_str = ' '.join(map(str, x))
                if x_str in data_strings:
                    continue
                data_strings.add(x_str)

                x_tf = self.fn_map[transform[0]](x, transform[1])

                flag = True
                if 'not' in subseq[0]:
                    flag = False
                x_subseq = self.fn_map[subseq[0].split()[-1]](x_tf, subseq[1], flag)
                if not x_subseq:  # Empty sequence
                    continue

                x_label = int(self.fn_map[labeling](x_subseq))
                if x_label in labels_enough_data:
                    continue

                if x_label not in data:
                    data[x_label] = []
                data[x_label].append(x)

                if len(data[x_label]) > per_class_data:
                    labels_enough_data.add(x_label)
                    if len(labels_enough_data) >= self.num_classes:
                        break

            assert len(labels_enough_data) == self.num_classes

            num_train_per_class = num_train // self.num_classes
            num_test_per_class = num_test // self.num_classes

            if uniform_classes:
                assert num_train_per_class * self.num_classes == num_train

            label_map = dict(zip(list(labels_enough_data), list(range(self.num_classes))))

            train, val, test = [], [], []
            
            for label in labels_enough_data:
                assert len(data[label]) >= num_train_per_class + 2 * num_test_per_class
                rng.shuffle(data[label])

                train_instances = data[label][:num_train_per_class]
                val_instances = data[label][num_train_per_class+1:num_train_per_class+num_test_per_class]
                test_instances = data[label][-num_test_per_class:]

                for instance in train_instances:
                    train.append((instance, label_map[label]))
                for instance in val_instances:
                    val.append((instance, label_map[label]))
                for instance in test_instances:
                    test.append((instance, label_map[label]))

            rng.shuffle(train)

            all_data.append((train, val, test))

        return all_data

    def task_description(self, transform, subseq, labeling):
        transform = map(str, transform)
        subseq = map(str, subseq)

        return ' -> '.join([' '.join(transform), ' '.join(subseq), labeling])

    def construct_tasks(self):

        transforms = []

        for k in [3, 5, 7]:
            transforms.append(('mul', k))
        for k in [0, 3, 5]:
            transforms.append(('add', k))
        for k in [2, 3]:
            transforms.append(('div', k))
        for k in [5, 6, 7]:
            transforms.append(('mod', k))

        subseq = []

        #for k in [2, 3, 4, 5]:
        for k in [2, 3]:
            subseq.append(('multiple', k))
            subseq.append(('not multiple', k))

        #for k in [8, 9, 10, 11, 12]:
        for k in [5, 6, 7]:
            subseq.append(('greater', k))
            subseq.append(('not greater', k))

        #for k in [2, 3, 4, 5, 6]:
        for k in [2, 3]:
            subseq.append(('divisors', k))
            subseq.append(('not divisors', k))

        labeling = [
            'count',
            'min',
            'max',
            'mean',
            'median',
            'mode',
            'first',
            'last',
            'middle',
            'max-min'
        ]

        self.transforms = transforms
        self.subseq = subseq
        self.labeling = labeling

    def function_mapping(self):

        def mul(x, k):
            return [(k * e) % self.vocab_size for e in x]

        def add(x, k):
            return [(k + e) % self.vocab_size for e in x]

        def div(x, k):
            return [e // k for e in x]

        def mod(x, k):
            return [e % k for e in x]

        def multiple(x, k, flag):
            if flag:
                return [e for e in x if e % k == 0]
            else:
                return [e for e in x if e % k != 0]

        def greater(x, k, flag):
            if flag:
                return [e for e in x if e > k]
            else:
                return [e for e in x if e <= k]

        def divisors(x, k, flag):
            if flag:
                return [e for e in x if self.num_divisors[e - 1] == k]
            else:
                return [e for e in x if self.num_divisors[e - 1] != k]

        def count(x):
            return len(x)

        def first(x):
            return x[0]

        def last(x):
            return x[-1]

        def middle(x):
            return x[len(x) // 2]

        def maxmindiff(x):
            return max(x) - min(x)

        def mode(x):
            return stats.mode(x)[0][0]

        fn_map = {
            'mul': mul,
            'add': add,
            'div': div,
            'multiple': multiple,
            'greater': greater,
            'divisors': divisors,
            'min': min,
            'max': max,
            'mean': np.mean,
            'median': np.median,
            'mode': mode,
            'max-min': maxmindiff,
            'count': len,
            'first': first,
            'last': last,
            'middle': middle,
            'mod': mod
        }

        return fn_map

#max_tasks = 650
#max_samples = 500
#seqlen = 5
#vocab_size = 12
#num_classes = 4
#logdir = '/tmp/tasks.txt'
#
#task_generator = TaskGenerator(max_tasks, max_samples, seqlen, vocab_size, num_classes, logdir)
##tasks = task_generator.generate_tasks()
#load_tasks = task_generator.load_tasks(logdir)
##data = task_generator.generate_data(load_tasks)
#data = task_generator.generate_data(load_tasks[:64], 10000)
