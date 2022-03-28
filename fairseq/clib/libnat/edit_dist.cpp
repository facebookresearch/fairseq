/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h> // @manual=//caffe2:torch_extension
#include <algorithm>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

using namespace ::std;

vector<vector<uint32_t>> edit_distance2_with_dp(
    vector<uint32_t>& x,
    vector<uint32_t>& y) {
  uint32_t lx = x.size();
  uint32_t ly = y.size();
  vector<vector<uint32_t>> d(lx + 1, vector<uint32_t>(ly + 1));
  for (uint32_t i = 0; i < lx + 1; i++) {
    d[i][0] = i;
  }
  for (uint32_t j = 0; j < ly + 1; j++) {
    d[0][j] = j;
  }
  for (uint32_t i = 1; i < lx + 1; i++) {
    for (uint32_t j = 1; j < ly + 1; j++) {
      d[i][j] =
          min(min(d[i - 1][j], d[i][j - 1]) + 1,
              d[i - 1][j - 1] + 2 * (x.at(i - 1) == y.at(j - 1) ? 0 : 1));
    }
  }
  return d;
}

vector<vector<uint32_t>> edit_distance2_backtracking(
    vector<vector<uint32_t>>& d,
    vector<uint32_t>& x,
    vector<uint32_t>& y,
    uint32_t terminal_symbol) {
  vector<uint32_t> seq;
  vector<vector<uint32_t>> edit_seqs(x.size() + 2, vector<uint32_t>());
  /*
  edit_seqs:
  0~x.size() cell is the insertion sequences
  last cell is the delete sequence
  */

  if (x.size() == 0) {
    edit_seqs.at(0) = y;
    return edit_seqs;
  }

  uint32_t i = d.size() - 1;
  uint32_t j = d.at(0).size() - 1;

  while ((i >= 0) && (j >= 0)) {
    if ((i == 0) && (j == 0)) {
      break;
    }

    if ((j > 0) && (d.at(i).at(j - 1) < d.at(i).at(j))) {
      seq.push_back(1); // insert
      seq.push_back(y.at(j - 1));
      j--;
    } else if ((i > 0) && (d.at(i - 1).at(j) < d.at(i).at(j))) {
      seq.push_back(2); // delete
      seq.push_back(x.at(i - 1));
      i--;
    } else {
      seq.push_back(3); // keep
      seq.push_back(x.at(i - 1));
      i--;
      j--;
    }
  }

  uint32_t prev_op, op, s, word;
  prev_op = 0, s = 0;
  for (uint32_t k = 0; k < seq.size() / 2; k++) {
    op = seq.at(seq.size() - 2 * k - 2);
    word = seq.at(seq.size() - 2 * k - 1);
    if (prev_op != 1) {
      s++;
    }
    if (op == 1) // insert
    {
      edit_seqs.at(s - 1).push_back(word);
    } else if (op == 2) // delete
    {
      edit_seqs.at(x.size() + 1).push_back(1);
    } else {
      edit_seqs.at(x.size() + 1).push_back(0);
    }

    prev_op = op;
  }

  for (uint32_t k = 0; k < edit_seqs.size(); k++) {
    if (edit_seqs[k].size() == 0) {
      edit_seqs[k].push_back(terminal_symbol);
    }
  }
  return edit_seqs;
}

vector<vector<uint32_t>> edit_distance2_backtracking_with_delete(
    vector<vector<uint32_t>>& d,
    vector<uint32_t>& x,
    vector<uint32_t>& y,
    uint32_t terminal_symbol,
    uint32_t deletion_symbol) {
  vector<uint32_t> seq;
  vector<vector<uint32_t>> edit_seqs(x.size() + 1, vector<uint32_t>());
  /*
  edit_seqs:
  0~x.size() cell is the insertion sequences
  last cell is the delete sequence
  */

  if (x.size() == 0) {
    edit_seqs.at(0) = y;
    return edit_seqs;
  }

  uint32_t i = d.size() - 1;
  uint32_t j = d.at(0).size() - 1;

  while ((i >= 0) && (j >= 0)) {
    if ((i == 0) && (j == 0)) {
      break;
    }

    if ((j > 0) && (d.at(i).at(j - 1) < d.at(i).at(j))) {
      seq.push_back(1); // insert
      seq.push_back(y.at(j - 1));
      j--;
    } else if ((i > 0) && (d.at(i - 1).at(j) < d.at(i).at(j))) {
      seq.push_back(2); // delete
      seq.push_back(x.at(i - 1));
      i--;
    } else {
      seq.push_back(3); // keep
      seq.push_back(x.at(i - 1));
      i--;
      j--;
    }
  }

  uint32_t prev_op, op, s, word;
  prev_op = 0, s = 0;
  for (uint32_t k = 0; k < seq.size() / 2; k++) {
    op = seq.at(seq.size() - 2 * k - 2);
    word = seq.at(seq.size() - 2 * k - 1);
    if (prev_op != 1) {
      s++;
    }
    if (op == 1) // insert
    {
      edit_seqs.at(s - 1).push_back(word);
    } else if (op == 2) // delete
    {
      edit_seqs.at(s - 1).push_back(deletion_symbol);
    }

    prev_op = op;
  }

  for (uint32_t k = 0; k < edit_seqs.size(); k++) {
    if (edit_seqs.at(k).size() == 0) {
      edit_seqs.at(k).push_back(terminal_symbol);
    }
  }
  return edit_seqs;
}

vector<uint32_t> compute_ed2(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys) {
  vector<uint32_t> distances(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    distances.at(i) = d.at(xs.at(i).size()).at(ys.at(i).size());
  }
  return distances;
}

vector<vector<vector<uint32_t>>> suggested_ed2_path(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys,
    uint32_t terminal_symbol) {
  vector<vector<vector<uint32_t>>> seq(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    seq.at(i) =
        edit_distance2_backtracking(d, xs.at(i), ys.at(i), terminal_symbol);
  }
  return seq;
}

vector<vector<vector<uint32_t>>> suggested_ed2_path_with_delete(
    vector<vector<uint32_t>>& xs,
    vector<vector<uint32_t>>& ys,
    uint32_t terminal_symbol,
    uint32_t deletion_symbol) {
  vector<vector<vector<uint32_t>>> seq(xs.size());
  for (uint32_t i = 0; i < xs.size(); i++) {
    vector<vector<uint32_t>> d = edit_distance2_with_dp(xs.at(i), ys.at(i));
    seq.at(i) = edit_distance2_backtracking_with_delete(
        d, xs.at(i), ys.at(i), terminal_symbol, deletion_symbol);
  }
  return seq;
}

PYBIND11_MODULE(libnat, m) {
  m.def("compute_ed2", &compute_ed2, "compute_ed2");
  m.def("suggested_ed2_path", &suggested_ed2_path, "suggested_ed2_path");
  m.def(
      "suggested_ed2_path_with_delete",
      &suggested_ed2_path_with_delete,
      "suggested_ed2_path_with_delete");
}
