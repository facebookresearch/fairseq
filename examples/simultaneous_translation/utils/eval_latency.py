from latency import LatencyInference
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--start-from-zero", action="store_true")
    args = parser.parse_args()

    scorer = LatencyInference()
    recorder = []
    with open(args.input, 'r') as f:
        for line in f:
            steps = [
                int(x) - int(not args.start_from_zero)  
                for x in line.split(",")
            ]
            src_len = max(steps)
            
            steps = torch.LongTensor(steps).unsqueeze(0)
            src_len = torch.LongTensor([src_len]).unsqueeze(0)

            recorder.append(scorer(steps, src_len))
    
    average_results = {}
    for metric in [
        'differentiable_average_lagging', 
        'average_lagging', 
        'average_proportion'
        ]:
        average_results[metric] = sum(
            [x[metric][0, 0].item() for x in recorder]
        ) / len(recorder)
        print(f"{metric}: {average_results[metric]}")