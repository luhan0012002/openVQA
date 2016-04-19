require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

Utils = {}

function Utils.getNextBatch(ds, n, inputs)
    local q_words, a_words, targets = torch.Tensor(), torch.Tensor(), torch.Tensor() 
    local start_idx = (n-1) * batchSize+1
	local end_idx = n * batchSize
	if end_idx > ds.size then
		end_idx = ds.size
	end

    -- inputs{q_words, a_words, fc7, conv4, targets}  
    q_words:index(ds.input_q, 1, ds.indices:sub(start_idx, end_idx))
    table.insert(inputs, q_words:cuda())
    a_words:index(ds.input_a, 1, ds.indices:sub(start_idx, end_idx))
    table.insert(inputs, a_words:cuda())
    table.insert(inputs, getData.getBatchFc7(ds, ds.indices, start_idx, end_idx):cuda())
    local conv4 = getData.getBatchConv4(ds, ds.indices, start_idx, end_idx)
    conv4 = conv4/torch.max(conv4)
    table.insert(inputs, conv4:cuda())
    targets:index(ds.target, 1, ds.indices:sub(start_idx, end_idx))
    table.insert(inputs, targets:cuda())
    table.insert(inputs, ds.question[start_idx])
    table.insert(inputs, ds.answer[start_idx])
end

function Utils.loadData(split, isShuffle)
	local ds = getData.read(split, rho)
	ds.input_q = ds.input_q
	ds.input_a = ds.input_a
	ds.target = ds.target
	ds.img_id = ds.img_id

	local indices = torch.LongTensor(ds.size)
	
	if isShuffle then
		indices:randperm(ds.size)
	else
		for i = 1,indices:size(1) do
			indices[i] = i 
		end
	end	

	ds.indices = indices
	return ds
end


return Utils
