import torch

from mmpt.models import MMPTModel


model, tokenizer, aligner = MMPTModel.from_pretrained(
    "projects/retri/videoclip/how2.yaml")

model.eval()


# B, T, FPS, H, W, C (VideoCLIP is trained on 30 fps of s3d)
video_frames = torch.randn(1, 2, 30, 224, 224, 3)
caps, cmasks = aligner._build_text_seq(
    tokenizer("some text", add_special_tokens=False)["input_ids"]
)

caps, cmasks = caps[None, :], cmasks[None, :]  # bsz=1

with torch.no_grad():
    output = model(video_frames, caps, cmasks, return_score=True)
print(output["score"])  # dot-product