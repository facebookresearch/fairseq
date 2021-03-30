/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
C++ code for solving the linear assignment problem.
Based on the Auction Algorithm from https://dspace.mit.edu/bitstream/handle/1721.1/3265/P-2108-26912652.pdf and the implementation from: 
https://github.com/bkj/auction-lap
Adapted to be more efficient when each worker is looking for k jobs instead of 1.
*/
#include <torch/extension.h>
#include <iostream>
using namespace torch::indexing;
torch::Tensor balanced_assignment(torch::Tensor job_and_worker_to_score) {
    int max_iterations = 100;
    torch::Tensor epsilon = (job_and_worker_to_score.max() - job_and_worker_to_score.min()) / 50;
    epsilon.clamp_min_(1e-04);
    torch::Tensor worker_and_job_to_score = job_and_worker_to_score.detach().transpose(0,1).contiguous();
    int num_workers = worker_and_job_to_score.size(0);
    int num_jobs = worker_and_job_to_score.size(1);
    auto device = worker_and_job_to_score.device();
    int jobs_per_worker = num_jobs / num_workers;
    torch::Tensor value = worker_and_job_to_score.clone();
    int counter = 0;
    torch::Tensor max_value = worker_and_job_to_score.max();

    torch::Tensor bid_indices;
    torch::Tensor cost = worker_and_job_to_score.new_zeros({1, num_jobs});
    torch::Tensor bids = worker_and_job_to_score.new_empty({num_workers, num_jobs});
    torch::Tensor bid_increments = worker_and_job_to_score.new_empty({num_workers, jobs_per_worker});
    torch::Tensor top_values = worker_and_job_to_score.new_empty({num_workers, jobs_per_worker + 1});
    torch::Tensor high_bids = worker_and_job_to_score.new_empty({num_jobs});

    torch::Tensor top_index = top_values.to(torch::kLong);
    torch::Tensor high_bidders = top_index.new_empty({num_jobs});
    torch::Tensor have_bids = high_bidders.to(torch::kBool);
    torch::Tensor jobs_indices = torch::arange({num_jobs}, torch::dtype(torch::kLong).device(device));
    torch::Tensor true_tensor = torch::ones({1}, torch::dtype(torch::kBool).device(device));

    while (true) {
        bids.zero_();
        torch::topk_out(top_values, top_index, value, jobs_per_worker + 1, 1);

        // Each worker bids the difference in value between that job and the k+1th job
        torch::sub_out(bid_increments,
                       top_values.index({Slice(None, None), Slice(0, jobs_per_worker)}),
                       top_values.index({Slice(None, None), jobs_per_worker}).unsqueeze(1));

        bid_increments.add_(epsilon);
        bids.scatter_(1,
            top_index.index({Slice(None, None),Slice(0, jobs_per_worker)}),
            bid_increments);

        if (counter < max_iterations && counter > 0) {
            // Put in a minimal bid to retain items from the last round if no-one else bids for them this round
            bids.view(-1).index_put_({bid_indices}, epsilon);
        }

        // Find the highest bidding worker per job
        torch::max_out(high_bids, high_bidders, bids, 0);
        torch::gt_out(have_bids, high_bids, 0);

        if (have_bids.all().item<bool>()) {
            // All jobs were bid for
            break;
        }

        // Make popular items more expensive
        cost.add_(high_bids);
        torch::sub_out(value, worker_and_job_to_score, cost);

        bid_indices = ((high_bidders * num_jobs) + jobs_indices).index({have_bids});

        if (counter < max_iterations) {
            // Make sure that this item will be in the winning worker's top-k next time.
            value.view(-1).index_put_({bid_indices}, max_value);
        }
        else {
            // Suboptimal approximation that converges quickly from current solution
            value.view(-1).index_put_({bid_indices}, worker_and_job_to_score.view(-1).index({bid_indices}));
        }

        counter += 1;
    }

    return top_index.index({Slice(None, None), Slice(0, jobs_per_worker)}).reshape(-1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("balanced_assignment", &balanced_assignment, "Balanced Assignment");
}
