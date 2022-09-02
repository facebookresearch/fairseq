# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import json
import numpy as np
import torch
import pickle
import math

from tqdm import tqdm


class Predictor(object):
    """this base class is used to save predictions to disk
        (and being called by a evaluator later).
        Predictor has minimum support of single gpu prediction.
    """
    def __init__(self, config):
        self.pred_dir = None  # on-the-fly eval does not save the results.
        if hasattr(config, "eval") and config.eval is not None:
            self.pred_dir = config.eval.save_path
            os.makedirs(self.pred_dir, exist_ok=True)

    def __call__(self, outputs):
        """extract the prediction and save it."""
        raise NotImplementedError

    def predict_loop(self, model, eval_dataloader, output_file=None):
        """on-the-fly prediction on a single gpu."""
        self.full_scores = []
        model.eval()
        model = model.to(0)
        with torch.no_grad():
            for data in eval_dataloader:
                data = self.to_ctx(data)
                outputs = model(**data)
                outputs.update(data)
                self(outputs)
        return self.finalize(output_file)

    def finalize(self, output_file):
        pass

    def to_ctx(self, data, ctx=0, dtype=None):
        if isinstance(data, dict):
            for key in data:
                if torch.is_tensor(data[key]):
                    if dtype is not None and data[key].dtype == torch.float32:
                        data[key] = data[key].to(dtype)
                    data[key] = data[key].to(ctx)
            return data
        else:
            raise ValueError("non-dict type of batch is not supported yet.")


class NLGPredictor(Predictor):
    """Predicting Text from MMFusion models."""
    """TODO: make a context."""
    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.dataset.bert_name,
            bos_token="[CLS]", eos_token="[SEP]")
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id

    def predict_loop(self, model, eval_dataloader, output_file=None):
        """TODO: refactor base classes."""
        ctx = 0
        outputs = {"outputs": [], "targets": [[]]}
        model.eval()
        model = model.to(ctx)
        with torch.no_grad():
            for data in tqdm(eval_dataloader):
                data = self.to_ctx(data, ctx)
                self(data, model, outputs)
        return self.finalize(outputs, output_file)

    def __call__(self, data, model, outputs):
        data.update({
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id
        })

        output = model.generate(**data)
        assert len(output) == len(data["ref"])
        for idx, _output in enumerate(output):
            generated_text = self.tokenizer.decode(
                _output, skip_special_tokens=True)
            if generated_text == "":
                generated_text = "none"
            outputs["outputs"].append(generated_text)
            outputs["targets"][0].append(data["ref"][idx])
            if random.random() < 0.001:
                print("_output", _output)
                print("generated_text", generated_text)
                print("ref", data["ref"][idx])

    def finalize(self, outputs, output_file=None):
        if output_file is not None:
            with open(os.path.join(
                    self.pred_dir, output_file + ".json"), "w") as fw:
                json.dump(outputs, fw, indent=4)
        return outputs


class RetrievalPredictor(Predictor):
    """generated `pooled_video` and `pooled_text`."""
    def __init__(self, config):
        super().__init__(config)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.dataset.bert_name)

    def predict_loop(
        self,
        model,
        eval_dataloader,
        output_file="retrieval.npy"
    ):
        """on-the-fly prediction on a single gpu."""
        full_scores = []
        texts = []
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            for data in eval_dataloader:
                # convert to dict.
                if not isinstance(data, dict):
                    data = {
                        "caps": data[0],
                        "cmasks": data[1],
                        "vfeats": data[2],
                        "vmasks": data[3],
                        "video_id": data[4]
                    }
                data = self.to_ctx(data)
                outputs = model(**data)
                outputs.update(data)
                self(outputs, full_scores)
                for _cap in data["caps"]:
                    texts.append(
                        self.tokenizer.decode(_cap, skip_special_tokens=True)
                    )

        return self.finalize(full_scores, texts, output_file)

    def __call__(self, sample, full_scores):
        scores = self._get_pooled_outputs(sample)
        self._append_scores(scores, full_scores)

    def finalize(self, full_scores, texts, output_file=None):
        outputs = self._aggregate_scores(full_scores)
        if output_file is not None:
            np.save(os.path.join(self.pred_dir, output_file + ".npy"), outputs)
        return {"outputs": outputs, "texts": texts}

    def _get_pooled_outputs(self, outputs):
        if "pooled_video" in outputs:
            return outputs["pooled_video"], outputs["pooled_text"]
        else:
            raise ValueError("unknown format of outputs.")

    def _append_scores(self, scores, full_scores):
        assert len(scores) == 2
        if len(full_scores) == 0:
            full_scores.append([])
            full_scores.append([])
        full_scores[0].append(scores[0].cpu().detach().numpy())
        full_scores[1].append(scores[1].cpu().detach().numpy())

    def _aggregate_scores(self, scores):
        assert len(scores) == 2
        video_hidden = np.concatenate(scores[0], axis=0)
        text_hidden = np.concatenate(scores[1], axis=0)
        # clear up.
        self.full_scores = []
        return np.matmul(text_hidden, video_hidden.T)


class QAPredictor(Predictor):
    """generated `pooled_video` and `pooled_text`."""
    def __init__(self, config):
        super().__init__(config)
        """predictor maintains scores and aggregate them."""

    def predict_loop(self, model, eval_dataloader, output_file="qa.npy"):
        """on-the-fly prediction on a single gpu."""
        self.full_scores = []
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            for data in eval_dataloader:
                # reshape ans and dup video 5 times.
                v_len = data["vfeats"].size(1)
                hidden_size = data["vfeats"].size(2)
                data["vfeats"] = data["vfeats"].unsqueeze(1).repeat(1, 5, 1, 1).view(-1, v_len, hidden_size)
                data["vmasks"] = data["vmasks"].unsqueeze(1).repeat(1, 5, 1).view(-1, v_len)

                t_len = data["caps"].size(-1)
                data["caps"] = data["caps"].view(-1, t_len)
                data["cmasks"] = data["cmasks"].view(-1, t_len)

                data = self.to_ctx(data)
                outputs = model(**data)
                outputs.update(data)
                self(outputs)
        return self.finalize(output_file)

    def __call__(self, sample):
        hidden_size = sample["pooled_video"].size(-1)
        pooled_video = sample["pooled_video"].view(-1, 5, hidden_size)
        pooled_text = sample["pooled_text"].view(-1, 5, hidden_size)
        scores = torch.bmm(pooled_video, pooled_text.transpose(2, 1))
        scores = scores.argmax(-1)
        self._append_scores(scores[:, 0], sample["answers"], self.full_scores)

    def finalize(self, output_file=None):
        outputs, targets = self._aggregate_scores(self.full_scores)
        if output_file is not None:
            np.save(os.path.join(self.pred_dir, output_file + ".npy"), outputs)
        return {"outputs": outputs, "targets": targets}

    def _append_scores(self, scores, answers, full_scores):
        if len(full_scores) == 0:
            full_scores.append([])
            full_scores.append([])
        full_scores[0].append(scores.cpu().detach().numpy())
        full_scores[1].append(answers.cpu().detach().numpy())

    def _aggregate_scores(self, scores):
        assert len(scores) == 2
        outputs = np.concatenate(scores[0], axis=0)
        targets = np.concatenate(scores[1], axis=0)
        # clear up.
        self.full_scores = []
        return outputs, targets


class CrossTaskPredictor(Predictor):
    """
    CrossTaskPredictor needs to compute the average of logits
    for overlapped sliding-window.
    """
    def __init__(self, config):
        super().__init__(config)
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.max_video_len = config.dataset.max_video_len
        self.sliding_window = config.dataset.sliding_window
        self.sliding_window_size = config.dataset.sliding_window_size
        self.annotation_path = config.dataset.annotation_path

    def predict_loop(self, model, eval_dataloader, output_file="result.pkl"):
        """refactored from line 144:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py
        """
        ctx = 0
        model.eval()
        model = model.to(ctx)
        # this is not a loss but just compute neg_log_prob.
        Y_pred = {}
        Y_true = {}
        with torch.no_grad():
            for batch in eval_dataloader:
                self(batch, model, Y_pred, Y_true)
        return self.finalize(Y_pred, Y_true, output_file)

    def __call__(self, sample, model, Y_pred, Y_true):
        # please install dp from `https://github.com/DmZhukov/CrossTask`
        from dp import dp
        vid, task = sample['video_id'][0], sample['task'][0]
        sample = self.to_ctx(sample)
        # compute the average logits over sliding windows.
        output = model(**sample)
        batch_logits = output["logits"].cpu()

        video_len = sample["video_len"][0]

        # the following version is slow.
        logits = torch.zeros((video_len, batch_logits.size(1)))
        logits_counts = torch.zeros((video_len, 1), dtype=torch.long)
        # use the same loop as aligner to recover.
        batch_logit_idx = 0
        for window_start in range(0, video_len, self.sliding_window):
            video_end = min(video_len - window_start, self.sliding_window_size)
            logits[window_start: window_start + video_end] += batch_logits[
                batch_logit_idx: batch_logit_idx + video_end]
            batch_logit_idx += video_end
            logits_counts[window_start: window_start + video_end] += torch.ones((video_end, 1), dtype=torch.long)

            if (video_len - window_start) <= self.sliding_window_size:
                break

        logits /= logits_counts
        assert logits.size() == (video_len, batch_logits.size(1)), "{}, {}".format(logits.size(), video_len)

        O = self.lsm(logits)
        y = np.zeros(O.size(), dtype=np.float32)
        dp(y, -O.detach().cpu().numpy())
        if task not in Y_pred:
            Y_pred[task] = {}
        Y_pred[task][vid] = y
        annot_path = os.path.join(
            self.annotation_path, task+'_'+vid+'.csv')
        if os.path.exists(annot_path):
            if task not in Y_true:
                Y_true[task] = {}
            Y_true[task][vid] = self._read_assignment(
                *y.shape, annot_path)

    def finalize(self, Y_pred, Y_true, output_file=None):
        if output_file is not None:
            with open(
                    os.path.join(self.pred_dir, output_file + ".pkl"),
                    "wb") as fw:
                pickle.dump(
                    {"Y_pred": Y_pred, "Y_true": Y_true}, fw,
                    protocol=pickle.HIGHEST_PROTOCOL)
        return {"outputs": Y_pred, "targets": Y_true}

    def _read_assignment(self, T, K, path):
        """
        refactored from https://github.com/DmZhukov/CrossTask/blob/master/data.py
        Howto interpret contraints on loss that is going to be minimized:
        lambd is a big number;
        self.lambd * C is a big number for all valid position (csv stores invalids)

        def forward(self, O, Y, C):
            return (Y*(self.lambd * C - self.lsm(O))).mean(dim=0).sum()

        This will load the csv file and fill-in the step col from start to end rows.
        """

        Y = np.zeros([T, K], dtype=np.uint8)
        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                step = int(step) - 1
                Y[start:end, step] = 1
        return Y


class COINPredictor(Predictor):
    """
    COINPredictor is similar to CrossTask on sliding windows.
    """
    def __init__(self, config):
        super().__init__(config)
        self.max_video_len = config.dataset.max_video_len
        self.sliding_window = config.dataset.sliding_window
        self.sliding_window_size = config.dataset.sliding_window_size

    def predict_loop(self, model, eval_dataloader, output_file="result.pkl"):
        """refactored from line 144:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py
        """
        ctx = 0
        model.eval()
        model = model.to(ctx)
        # this is not a loss but just compute neg_log_prob.
        Y_pred = []
        Y_true = []
        with torch.no_grad():
            for batch in eval_dataloader:
                self(batch, model, Y_pred, Y_true)
        return self.finalize(Y_pred, Y_true, output_file)

    def __call__(self, sample, model, Y_pred, Y_true):
        sample = self.to_ctx(sample)
        # compute the average logits over sliding windows.
        output = model(**sample)
        logits = self._merge_windows(sample, output)
        Y_pred.append(logits.argmax(dim=1))
        Y_true.append(sample["video_targets"].squeeze(0).cpu())

    def _merge_windows(self, sample, output):
        targets = sample["targets"].reshape(-1).cpu()
        valid_mask = targets != -100
        targets = targets[valid_mask]
        batch_logits = output["logits"].cpu()
        batch_logits = batch_logits.reshape(-1, batch_logits.size(-1))
        batch_logits = batch_logits[valid_mask]

        video_len = sample["video_len"][0]

        # the following version is slow.
        logits = torch.zeros((video_len, batch_logits.size(1)))
        logits_counts = torch.zeros((video_len, 1), dtype=torch.long)
        # use the same loop as aligner to recover.
        batch_logit_idx = 0
        for window_start in range(0, video_len, self.sliding_window):
            video_end = min(video_len - window_start, self.sliding_window_size)
            logits[window_start: window_start + video_end] += batch_logits[
                batch_logit_idx: batch_logit_idx + video_end]
            batch_logit_idx += video_end
            logits_counts[window_start: window_start + video_end] += torch.ones((video_end, 1), dtype=torch.long)
            if (video_len - window_start) <= self.sliding_window_size:
                break
        logits /= logits_counts
        assert logits.size() == (video_len, batch_logits.size(1)), "{}, {}".format(logits.size(), video_len)
        return logits

    def finalize(self, Y_pred, Y_true, output_file=None):
        Y_pred = torch.cat(Y_pred, dim=0).numpy()
        Y_true = torch.cat(Y_true, dim=0).numpy()
        assert len(Y_pred) == len(Y_true)

        error_mask = Y_pred != Y_true
        print("sample error", Y_pred[error_mask][:10], Y_true[error_mask][:10])
        print("sample error", Y_pred[error_mask][10:20], Y_true[error_mask][10:20])

        if output_file is not None:
            with open(
                    os.path.join(self.pred_dir, output_file + ".pkl"),
                    "wb") as fw:
                pickle.dump(
                    {"Y_pred": Y_pred, "Y_true": Y_true}, fw,
                    protocol=pickle.HIGHEST_PROTOCOL)
        return {"outputs": Y_pred, "targets": Y_true}


class COINZSPredictor(COINPredictor):
    """
    COINZSPredictor for COIN zero-shot prediction.
    """

    def __init__(self, config):
        super().__init__(config)
        self.dataset_config = config.dataset

    def predict_loop(self, model, eval_dataloader, output_file="result.pkl"):
        """refactored from line 144:
        https://github.com/DmZhukov/CrossTask/blob/master/train.py
        """
        ctx = 0
        model.eval()
        model = model.to(ctx)

        with torch.no_grad():
            outputs = eval_dataloader.dataset.meta_processor.meta_text_labels(
                self.dataset_config)
            outputs = self.to_ctx(outputs, ctx)
            label_hidden_states = model.forward_text(**outputs).cpu()
            label_sim = label_hidden_states @ label_hidden_states.t()
            num_labels = label_sim.size(0)
            eye_mask = ~torch.eye(num_labels, dtype=torch.bool)
            label_sim = label_sim.masked_select(eye_mask).view(num_labels, num_labels - 1)
            lbd = label_sim.max()

        # this is not a loss but just compute neg_log_prob.
        Y_pred = []
        Y_true = []
        with torch.no_grad():
            for batch in eval_dataloader:
                self(batch, label_hidden_states, model, lbd, Y_pred, Y_true)
        return self.finalize(Y_pred, Y_true, output_file)

    def reshape_subsample(self, sample):
        for key in sample:
            if torch.is_tensor(sample[key]):
                sample[key] = self.flat_subsample(sample[key])
        return sample

    def flat_subsample(self, tensor):
        if len(tensor.size()) > 1 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def __call__(self, sample, label_hidden_states, model, lbd, Y_pred, Y_true):
        sample = self.reshape_subsample(sample)
        sample = self.to_ctx(sample)
        # compute the average logits over sliding windows.
        sample["output_hidden_states"] = True
        video_outputs = model.forward_video(**sample).cpu()
        output = {"logits": video_outputs[:, 1:sample["vmasks"].size(1)+1] @ label_hidden_states.t()}
        logits = self._merge_windows(sample, output)
        # logic of zero-shot for sequence labeling.
        logits_argmax = logits.argmax(dim=1) + 1  # 0 is "O" label.
        logits_max = logits.max(dim=1)[0]

        pred = torch.zeros_like(logits_argmax)
        label_select = logits_max > lbd  # 73 or 74
        pred[label_select] = logits_argmax[label_select]

        Y_pred.append(pred)
        Y_true.append(sample["video_targets"].squeeze(0).cpu())

    def finalize(self, Y_pred, Y_true, output_file=None):
        Y_pred = torch.cat(Y_pred, dim=0).numpy()
        Y_true = torch.cat(Y_true, dim=0).numpy()
        assert len(Y_pred) == len(Y_true)

        error_mask = Y_pred != Y_true
        print("sample error", Y_pred[error_mask][:10], Y_true[error_mask][:10])
        print("sample error", Y_pred[error_mask][10:20], Y_true[error_mask][10:20])

        if output_file is not None:
            with open(
                    os.path.join(self.pred_dir, output_file + ".pkl"),
                    "wb") as fw:
                pickle.dump(
                    {"Y_pred": Y_pred, "Y_true": Y_true}, fw,
                    protocol=pickle.HIGHEST_PROTOCOL)
        return {"outputs": Y_pred, "targets": Y_true}


class DiDeMoPredictor(Predictor):
    """reference: https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/eval.py
    https://github.com/LisaAnne/LocalizingMoments/blob/master/utils/data_processing.py
    """
    def __init__(self, config):
        super().__init__(config)
        # load targets.
        with open(config.dataset.test_path) as data_file:
            self.test_data = json.load(data_file)

    def predict_loop(self, model, eval_dataloader, output_file="didemo.npy"):
        """
        TODO: two solutions here.
        """
        import itertools
        # 21 chunks.
        self.possible_segments = [(0,0), (1,1), (2,2), (3,3), (4,4), (5,5)]
        for i in itertools.combinations(range(6), 2):
            self.possible_segments.append(i)
        # pick segments from a video.

        """on-the-fly prediction on a single gpu."""
        self.full_scores = []
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            for data in eval_dataloader:
                # TODO special forwarding logic here.
                data = self.to_ctx(data)
                data["output_hidden_states"] = True
                hidden_video = model.forward_video(**data)
                data["output_hidden_states"] = False
                pooled_text = model.forward_text(**data)
                outputs = {
                    "hidden_video": hidden_video,
                    "pooled_text": pooled_text
                }
                outputs.update(data)
                self(outputs)
        return self.finalize(output_file)

    def __call__(self, sample):
        # TODO: make an index select from self.possible_segments.
        hidden_video = sample["hidden_video"]
        pooled_text = sample["pooled_text"]
        vmasks = sample["vmasks"]
        # probably maintain valid results here.

        hidden_video = hidden_video[:, 1:-1, :]
        # probably maintain valid results here.
        pooled_video = []
        for s, e in self.possible_segments:
            pooled_video.append(
                torch.mean(
                    hidden_video[:, int(s*5):int((e+1)*5), :],
                    dim=1, keepdim=True)
            )
        pooled_video = torch.cat(pooled_video, dim=1)
        scores = torch.bmm(
            pooled_video, pooled_text.unsqueeze(-1)).squeeze(-1).cpu()

        ranks = scores.argsort(dim=-1, descending=True)

        for batch_idx, rank in enumerate(ranks):
            rank_of_moment = []
            for m_idx, moment in enumerate(rank):
                s, e = self.possible_segments[moment.item()]
                if torch.any(
                    vmasks[batch_idx, int(s*5):int((e+1)*5)]
                ):
                    rank_of_moment.append((s, e))
            self.full_scores.append(rank_of_moment)

    def finalize(self, output_file=None):
        outputs = self._aggregate_scores(self.full_scores)
        if output_file is not None:
            np.save(os.path.join(self.pred_dir, output_file + ".npy"), outputs)
        return {"outputs": outputs, "targets": self.test_data}

    def _aggregate_scores(self, scores):
        self.full_scores = []
        return scores
