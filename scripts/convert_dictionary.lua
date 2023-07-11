-- Copyright (c) Facebook, Inc. and its affiliates.
--
-- This source code is licensed under the MIT license found in the
-- LICENSE file in the root directory of this source tree.
--
-- Usage: convert_dictionary.lua <dict.th7>
require 'fairseq'
require 'torch'
require 'paths'

if #arg < 1 then
   print('usage: convert_dictionary.lua <dict.th7>')
   os.exit(1)
end
if not paths.filep(arg[1]) then
   print('error: file does not exit: ' .. arg[1])
   os.exit(1)
end

dict = torch.load(arg[1])
dst = paths.basename(arg[1]):gsub('.th7', '.txt')
assert(dst:match('.txt$'))

f = io.open(dst, 'w')
for idx, symbol in ipairs(dict.index_to_symbol) do
  if idx > dict.cutoff then
    break
  end
  f:write(symbol)
  f:write(' ')
  f:write(dict.index_to_freq[idx])
  f:write('\n')
end
f:close()
