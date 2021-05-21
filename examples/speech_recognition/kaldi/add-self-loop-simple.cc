/*
* Copyright (c) Facebook, Inc. and its affiliates.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
*/

#include <iostream>
#include "fstext/fstext-lib.h" // @manual
#include "util/common-utils.h" // @manual

/*
 * This program is to modify a FST without self-loop by:
 *   for each incoming arc with non-eps input symbol, add a self-loop arc
 *   with that non-eps symbol as input and eps as output.
 *
 * This is to make sure the resultant FST can do deduplication for repeated
 * symbols, which is very common in acoustic model
 *
 */
namespace {
int32 AddSelfLoopsSimple(fst::StdVectorFst* fst) {
  typedef fst::MutableArcIterator<fst::StdVectorFst> IterType;

  int32 num_states_before = fst->NumStates();
  fst::MakePrecedingInputSymbolsSame(false, fst);
  int32 num_states_after = fst->NumStates();
  KALDI_LOG << "There are " << num_states_before
            << " states in the original FST; "
            << " after MakePrecedingInputSymbolsSame, there are "
            << num_states_after << " states " << std::endl;

  auto weight_one = fst::StdArc::Weight::One();

  int32 num_arc_added = 0;

  fst::StdArc self_loop_arc;
  self_loop_arc.weight = weight_one;

  int32 num_states = fst->NumStates();
  std::vector<std::set<int32>> incoming_non_eps_label_per_state(num_states);

  for (int32 state = 0; state < num_states; state++) {
    for (IterType aiter(fst, state); !aiter.Done(); aiter.Next()) {
      fst::StdArc arc(aiter.Value());
      if (arc.ilabel != 0) {
        incoming_non_eps_label_per_state[arc.nextstate].insert(arc.ilabel);
      }
    }
  }

  for (int32 state = 0; state < num_states; state++) {
    if (!incoming_non_eps_label_per_state[state].empty()) {
      auto& ilabel_set = incoming_non_eps_label_per_state[state];
      for (auto it = ilabel_set.begin(); it != ilabel_set.end(); it++) {
        self_loop_arc.ilabel = *it;
        self_loop_arc.olabel = 0;
        self_loop_arc.nextstate = state;
        fst->AddArc(state, self_loop_arc);
        num_arc_added++;
      }
    }
  }
  return num_arc_added;
}

void print_usage() {
  std::cout << "add-self-loop-simple usage:\n"
               "\tadd-self-loop-simple <in-fst> <out-fst> \n";
}
} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    print_usage();
    exit(1);
  }

  auto input = argv[1];
  auto output = argv[2];

  auto fst = fst::ReadFstKaldi(input);
  auto num_states = fst->NumStates();
  KALDI_LOG << "Loading FST from " << input << " with " << num_states
            << " states." << std::endl;

  int32 num_arc_added = AddSelfLoopsSimple(fst);
  KALDI_LOG << "Adding " << num_arc_added << " self-loop arcs " << std::endl;

  fst::WriteFstKaldi(*fst, std::string(output));
  KALDI_LOG << "Writing FST to " << output << std::endl;

  delete fst;
}