-- Copyright (c) Facebook, Inc. and its affiliates.
--
-- This source code is licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.
--
-- Usage: convert_model.lua <model_epoch1.th7>
require 'torch'
local fairseq = require 'fairseq'

model = torch.load(arg[1])

function find_weight_norm(container, module)
  for _, wn in ipairs(container:listModules()) do
    if torch.type(wn) == 'nn.WeightNorm' and wn.modules[1] == module then
      return wn
    end
  end
end

function push_state(dict, key, module)
  if torch.type(module) == 'nn.Linear' then
    local wn = find_weight_norm(model.module, module)
    assert(wn)
    dict[key .. '.weight_v'] = wn.v:float()
    dict[key .. '.weight_g'] = wn.g:float()
  elseif torch.type(module) == 'nn.TemporalConvolutionTBC' then
    local wn = find_weight_norm(model.module, module)
    assert(wn)
    local v = wn.v:float():view(wn.viewOut):transpose(2, 3)
    dict[key .. '.weight_v'] = v
    dict[key .. '.weight_g'] = wn.g:float():view(module.weight:size(3), 1, 1)
  else
    dict[key .. '.weight'] = module.weight:float()
  end
  if module.bias then
    dict[key .. '.bias'] = module.bias:float()
  end
end

encoder_dict = {}
decoder_dict = {}
combined_dict = {}

function encoder_state(encoder)
  luts = encoder:findModules('nn.LookupTable')
  push_state(encoder_dict, 'embed_tokens', luts[1])
  push_state(encoder_dict, 'embed_positions', luts[2])

  fcs = encoder:findModules('nn.Linear')
  assert(#fcs >= 2)
  local nInputPlane = fcs[1].weight:size(1)
  push_state(encoder_dict, 'fc1', table.remove(fcs, 1))
  push_state(encoder_dict, 'fc2', table.remove(fcs, #fcs))

  for i, module in ipairs(encoder:findModules('nn.TemporalConvolutionTBC')) do
    push_state(encoder_dict, 'convolutions.' .. tostring(i - 1), module)
    if nInputPlane ~= module.weight:size(3) / 2 then
      push_state(encoder_dict, 'projections.' .. tostring(i - 1), table.remove(fcs, 1))
    end
    nInputPlane = module.weight:size(3) / 2
  end
  assert(#fcs == 0)
end

function decoder_state(decoder)
  luts = decoder:findModules('nn.LookupTable')
  push_state(decoder_dict, 'embed_tokens', luts[1])
  push_state(decoder_dict, 'embed_positions', luts[2])

  fcs = decoder:findModules('nn.Linear')
  local nInputPlane = fcs[1].weight:size(1)
  push_state(decoder_dict, 'fc1', table.remove(fcs, 1))
  push_state(decoder_dict, 'fc2', fcs[#fcs - 1])
  push_state(decoder_dict, 'fc3', fcs[#fcs])

  table.remove(fcs, #fcs)
  table.remove(fcs, #fcs)

  for i, module in ipairs(decoder:findModules('nn.TemporalConvolutionTBC')) do
    if nInputPlane ~= module.weight:size(3) / 2 then
      push_state(decoder_dict, 'projections.' .. tostring(i - 1), table.remove(fcs, 1))
    end
    nInputPlane = module.weight:size(3) / 2

    local prefix = 'attention.' .. tostring(i - 1)
    push_state(decoder_dict, prefix .. '.in_projection', table.remove(fcs, 1))
    push_state(decoder_dict, prefix .. '.out_projection', table.remove(fcs, 1))
    push_state(decoder_dict, 'convolutions.' .. tostring(i - 1), module)
  end
  assert(#fcs == 0)
end


_encoder = model.module.modules[2]
_decoder = model.module.modules[3]

encoder_state(_encoder)
decoder_state(_decoder)

for k, v in pairs(encoder_dict) do
  combined_dict['encoder.' .. k] = v
end
for k, v in pairs(decoder_dict) do
  combined_dict['decoder.' .. k] = v
end


torch.save('state_dict.t7', combined_dict)
