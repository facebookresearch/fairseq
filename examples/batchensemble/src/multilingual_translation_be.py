from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask


@register_task("multilingual_translation_be")
class MultilingualTranslationBatchEnsembleTask(MultilingualTranslationTask):
    """A task for multilingual translation with BatchEnsemble.
    """
    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        # fmt: off
        # args for Training with BatchEnsemble
        parser.add_argument('--batch-ensemble-vanilla', default=False, action='store_true',
                            help='Adjusts the behavior of BatchEnsemble to be like that of the paper, an ensemble')
        parser.add_argument('--batch-ensemble-root', type=int, default=-1,
                            help='BatchEnsemble root task (0-based) for lifelong learning')
        parser.add_argument('--batch-ensemble-linear-init', default=False, action='store_true',
                            help='Initialize weights and biases akin to nn.Linear')
        parser.add_argument('--batch-ensemble-lr-multiplier', type=float, default=1.0,
                            help='Learning rate multiplier for BatchEnsemble parameters')
        parser.add_argument('--batch-ensemble-lr-relative', default=False, action='store_true',
                            help='Learning rate relative to optimizer.get_lr()')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)

        self.lang_pairs = args.lang_pairs
        if isinstance(self.lang_pairs, str):
            self.lang_pairs = self.lang_pairs.split(",")
        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.n_tasks = len(self.lang_pairs)
        assert self.n_tasks > 0

        if training:
            # Validate argument batch_ensemble_root
            assert args.batch_ensemble_root == -1 or (
                args.batch_ensemble_root >= 0 and
                args.batch_ensemble_root < len(self.lang_pairs)
            )

            self.lr_multiplier = getattr(args, "batch_ensemble_lr_multiplier", 1.0)
            self.context_lr = self.args.lr[0] * self.lr_multiplier
            self.relative_lr = getattr(args, "batch_ensemble_lr_relative", False)
            # Tracking variable to only update LR once
            self.lr_fixed = False

    def _get_lang_pair_idx(self, lang_pair):
        # Set language pair index
        lang_pair_idx = [
            i
            for i, lp in enumerate(self.model_lang_pairs)
            if lp == lang_pair
        ]

        assert len(lang_pair_idx) == 1
        lang_pair_idx = lang_pair_idx[0]
        return lang_pair_idx

    def _per_lang_pair_train_loss(
        self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
    ):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        return super()._per_lang_pair_train_loss(
            lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        if self.relative_lr or not self.lr_fixed:
            self.lr_fixed = True

            if self.relative_lr:
                self.context_lr = optimizer.get_lr() * self.lr_multiplier

            # Update the state dictionary of the optimizer to reflect the
            # new learning rates for the specified parameters
            print("Fixing Optimizer Learning Rates")
            optimizer_state = optimizer.state_dict()
            for param_group in optimizer_state["param_groups"]:
                if "_name" in param_group and "context_param" in param_group["_name"]:
                    print(
                        "Adjusting ",
                        param_group["_name"],
                        " from ",
                        param_group["lr"],
                        " to ",
                        self.context_lr
                    )
                    param_group["lr"] = self.context_lr
            optimizer.load_state_dict(optimizer_state)

            print("New Optimizer State")
            optimizer_state = optimizer.state_dict()
            for param_group in optimizer_state["param_groups"]:
                if "_name" in param_group and "context_param" in param_group["_name"]:
                    print(param_group["_name"], "\t", param_group["lr"])

            import sys
            sys.stdout.flush()

        return super().train_step(
            sample, model, criterion, optimizer, update_num, ignore_grad
        )

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        model.models[lang_pair].decoder.set_lang_pair_idx(lang_pair_idx)

        return super()._per_lang_pair_valid_loss(
            lang_pair, model, criterion, sample
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        lang_pair = f"{self.args.source_lang}-{self.args.target_lang}"

        # Update language pair index
        lang_pair_idx = self._get_lang_pair_idx(lang_pair)
        for model in models:
            model.decoder.set_lang_pair_idx(lang_pair_idx)

        return super().inference_step(
            generator, models, sample, prefix_tokens, constraints
        )
