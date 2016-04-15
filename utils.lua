require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

Utils = {}

function Utils.getNextBatch(ds, indices, n, words, fc7, conv4, targets)
	local start_idx = (n-1) * batchSize+1
	local end_idx = n * batchSize
	if end_idx > ds.size then
		end_idx = ds.size
	end

	local inputs = {}
	words:index(ds.input, 1, indices:sub(start_idx, end_idx))
	targets:index(ds.target, 1, indices:sub(start_idx, end_idx))
	fc7 = getData.getBatchFc7(ds, indices, start_idx, end_idx)
	conv4 = getData.getBatchConv4(ds, indices, start_idx, end_idx)
	conv4 = conv4/torch.max(conv4)
	table.insert(inputs, words)
	table.insert(inputs, fc7)
	table.insert(inputs, conv4)
	return inputs
end

function Utils.loadData(split, isShuffle)
	local ds = getData.read(split, rho)
	ds.input = ds.input:cuda()
	ds.target = ds.target:cuda()
	ds.img_id = ds.img_id:cuda()

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
