import os
import torch
import numpy as np

# from eval_ipex_cnn_true import Seq2Seq, extract_fbank, eval
from openvino.runtime import CompiledModel 

import openvino.runtime as ov
from openvino.tools import mo
from openvino.runtime import Core, serialize
import nncf, sys
from torch.utils.data import Dataset, TensorDataset, DataLoader

ie = Core()

def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)
    
def transform_fn(data_item):
    src = data_item[0]
    return src


ir_enc_model_path = "onnx_models/encoder_xiping.xml"
ir_enc_int8_model_path  = "IR_INT8/encoder_int8.xml"
enc_model = ie.read_model(ir_enc_model_path)

# src = src.unsqueeze(0)
src = torch.randn((1, 128))
src = src.detach().numpy()
pot_dataset = TensorDataset(torch.from_numpy(src))
pot_dataloader = DataLoader(pot_dataset)

calibration_dataset = nncf.Dataset(pot_dataloader, transform_fn)

quantized_model = nncf.quantize(enc_model, calibration_dataset,
                            subset_size=1)
    
# int8_ir_path = model_path + model_name + "_int8" + ".xml"
ov.serialize(quantized_model, ir_enc_int8_model_path)
print("==NNCF Quantization Success==",ir_enc_int8_model_path)