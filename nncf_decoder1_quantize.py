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

class DecoderDataset(Dataset):
    def __init__(self, enc_model, dec_model, data_count=100):
        self.dataset = np.random.rand(data_count)
        self.enc_model = enc_model
        self.dec_model = dec_model
        self.enc_output_keys = list(self.enc_model.outputs)
        self.dec_output_keys = list(self.dec_model.outputs)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokens = torch.randn((1, 128))
        enc_out = self.enc_model(tokens)
        encoder_out = enc_out[self.enc_output_keys[0]]
        encoder_padding_mask = enc_out[self.enc_output_keys[1]]
        encoder_embedding = enc_out[self.enc_output_keys[2]]
        decoder_input = {"tokens": tokens,
                         "encoder_out":encoder_out,
                         "encoder_padding_mask": encoder_padding_mask}
        decoder_result = self.dec_model(decoder_input)
        output_keys = list(self.dec_model.outputs)

        x = decoder_result[output_keys[0]]
        attn = decoder_result[output_keys[1]]
        inner_states_0 = decoder_result[output_keys[2]]
        inner_states_1 = decoder_result[output_keys[3]]
        inner_states_2 = decoder_result[output_keys[4]]
        inner_states_3 = decoder_result[output_keys[5]]
        decoder_result_list = [torch.from_numpy(x),
                            {"attn":[torch.from_numpy(attn)],
                            "inner_states":[torch.from_numpy(inner_states_0), 
                                            torch.from_numpy(inner_states_1), 
                                            torch.from_numpy(inner_states_2), 
                                            torch.from_numpy(inner_states_3)]}]
        incremental_state={}
        # Past key value.
        for i in range(6):
            incremental_state["prev_key_"+str(i)] = decoder_result[output_keys[6+i*2]]
            incremental_state["prev_value_"+str(i)] = decoder_result[output_keys[6+i*2+1]]

        decoder1_input = {"tokens": tokens,
                         "encoder_padding_mask": encoder_padding_mask}
        for pk in list(incremental_state.keys()):
            decoder1_input[pk] = incremental_state[pk]

        print("==decoder1_input len==",len(decoder1_input))
        print("==decoder1_input[0].shape==",decoder1_input["tokens"].shape)
        print("==decoder1_input[prev_key_0].shape==",decoder1_input["prev_key_0"].shape)
        return decoder1_input

def print_model(model):
    model_size = sum(p.numel() for p in model.parameters()) / 1000000.
    print('Model size: %.2fM' % model_size)
    
def transform_fn(data_item):
    print("==data_item len==",len(data_item))
    print("==data_item[0].shape==",data_item["tokens"].shape)
    print("==data_item[prev_key_0].shape==",data_item["prev_key_0"].shape)
    inputs = {
        "tokens":data_item["tokens"][0], 
        "prev_key_0":data_item["prev_key_0"][0], 
        "prev_value_0":data_item["prev_value_0"][0],
        "prev_key_1":data_item["prev_key_1"][0], 
        "prev_value_1":data_item["prev_value_1"][0],
        "prev_key_2":data_item["prev_key_2"][0], 
        "prev_value_2":data_item["prev_value_2"][0],
        "prev_key_3":data_item["prev_key_3"][0], 
        "prev_value_3":data_item["prev_value_3"][0],
        "prev_key_4":data_item["prev_key_4"][0], 
        "prev_value_4":data_item["prev_value_4"][0],
        "prev_key_5":data_item["prev_key_5"][0], 
        "prev_value_5":data_item["prev_value_5"][0],
        "encoder_padding_mask":data_item["encoder_padding_mask"][0], 

    }
    return inputs


ir_enc_model_path = "onnx_models/encoder_xiping.xml"
enc_model = ie.read_model(ir_enc_model_path)

ir_dec0_model_path = "onnx_models/decoder0_xiping.xml"
ir_dec0_int8_model_path  = "IR_INT8/decoder0_int8.xml"
dec0_model = ie.read_model(ir_dec0_model_path)


ir_dec1_model_path = "onnx_models/decoder1_xiping.xml"
ir_dec1_int8_model_path  = "IR_INT8/decoder1_int8.xml"
dec1_model = ie.read_model(ir_dec1_model_path)

# src = src.unsqueeze(0)
src = torch.randn((1, 128))
tokens = torch.randn((1, 128))
# src = src.detach().numpy()
enc_compile_model = ie.compile_model(enc_model)
dec0_compile_model = ie.compile_model(dec0_model)

dataset = DecoderDataset(enc_compile_model, dec0_compile_model, data_count=100)
pot_dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

calibration_dataset = nncf.Dataset(pot_dataloader, transform_fn)

quantized_model = nncf.quantize(dec1_model, calibration_dataset, subset_size=1)
    
# int8_ir_path = model_path + model_name + "_int8" + ".xml"
ov.serialize(quantized_model, ir_dec1_int8_model_path)
print("==NNCF Quantization Success==",ir_dec1_int8_model_path)