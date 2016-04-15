require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

local model_utils = require 'model_utils'
local model = require 'model'
local Utils = require 'utils'
local Train = {}

function Train.foward(clones, protos, totalInput, targets, batchSize)
	fc7, words, conv4 = unpack(totalInput)
	ht = {}
	ct = {}
	rt = {}
	wordEmbed = {}
	imageEmbed = protos.imageEmbed:forward(fc7)
	table.insert(rt, clones['attention'][1]:forward({torch.CudaTensor(batchSize, hiddenSize):fill(0), conv4}))
	local tmp = clones['lstm'][1]:forward({imageEmbed, torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0), rt[1]})
	table.insert(ht, tmp[1])
	table.insert(ct, tmp[2])
	for t = 2, rho+1 do
		table.insert(wordEmbed, clones['wordEmbed'][t-1]:forward(words:select(2,t-1)))
		table.insert(rt, clones['attention'][t]:forward({ht[t-1], conv4}))
		local tmp = clones['lstm'][t]:forward({wordEmbed[t-1], ht[t-1], ct[t-1], rt[t]})
		table.insert(ht, tmp[1])
		table.insert(ct, tmp[2])
	end 
	outputs = protos.classify:forward({wordEmbed[rho], ht[rho+1], ct[rho+1], rt[rho+1]})
	err = protos.criterion:forward(outputs, targets)
	totalOutput = {ht, ct, rt, wordEmbed, imageEmbed, outputs}
	return totalOutput, err
end

function Train.backward(clones, protos, totalInput, targets, totalOutput, grad_params)
	fc7, words, conv4 = unpack(totalInput)
	ht, ct, rt, wordEmbed, imageEmbed, outputs = unpack(totalOutput)
	gradOutput = protos.criterion:backward(outputs, targets)
	prevGradInput = protos.classify:backward({wordEmbed[rho], ht[rho+1], ct[rho+1], rt[rho+1]}, gradOutput)
	for t = rho+1, 2, -1 do
		prevGradInput = clones['lstm'][t]:backward({wordEmbed[t-1], ht[t-1], ct[t-1], rt[t]}, {prevGradInput[2], prevGradInput[3]})
		clones['attention'][t]:backward({ht[t-1], conv4}, prevGradInput[4])
		clones['wordEmbed'][t-1]:backward(words:select(2, t-1), prevGradInput[1])
	end
	prevGradInput = clones['lstm'][1]:backward({imageEmbed, torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0), rt[1]}, {prevGradInput[2], prevGradInput[3]})
	clones['attention'][1]:backward({torch.CudaTensor(batchSize, hiddenSize):fill(0), conv4}, prevGradInput[4])
	protos.imageEmbed:backward(fc7, prevGradInput[1])
	grad_params:clamp(-5, 5)
end

function Train.train_sgd(protos, ds, ds_val, solver_params)
	local nBatches_val = math.ceil(ds_val.size/batchSize)
	local nBatches = math.ceil(ds_train.size/batchSize)
	-- local lstm, wordEmbed, imageEmbed, classify, criterion = model[0],model[1],model[2],model[3],model[4]
	local params, grad_params = model_utils.combine_all_parameters(protos.lstm, protos.wordEmbed, protos.imageEmbed, protos.classify, protos.attention)
	clones = {}
	clones['lstm'] = model_utils.clone_many_times(protos.lstm, rho+1, not protos.lstm.parameters)
	clones['wordEmbed'] = model_utils.clone_many_times(protos.wordEmbed, rho, not protos.wordEmbed.parameters) -- 1 unit less than lstm
	clones['attention'] = model_utils.clone_many_times(protos.attention, rho+1, not protos.attention.parameters) 
	
	
	for epoch = 1, nEpoch do
		local epoch_err = 0
		local sanity_check_err = 0
		for n = 1, nBatches do
			-- get next training/testing batch
			local words, fc7, conv4, targets = torch.LongTensor(),torch.LongTensor(),torch.LongTensor(), torch.LongTensor() 
			words, fc7, conv4, targets = words:cuda(), fc7:cuda(), conv4:cuda(), targets:cuda()
			local gradOutput = nil
			local gradInput = nil
			-- feval function for sgd solver
			local function feval(x)
				if x ~= params then
					params:copy(x)
				end
				grad_params:zero()

				-- get mini-batch
				local inputs = Utils.getNextBatch(ds, ds.indices, n, words, fc7, conv4, targets)
				--print(inputs)
				--words = inputs[1]
				fc7 = inputs[2]
				conv4 = inputs[3]
				local batchSize = words:size(1)
				local totalInput = {fc7, words, conv4}
				-- forward step
				local totalOutput, err = Train.foward(clones, protos, totalInput, targets, batchSize)
				epoch_err = epoch_err + err			
				-- backward step
				Train.backward(clones, protos, totalInput, targets, totalOutput, grad_params)

				return err, grad_params
			end

			local _, fs = optim.sgd(feval, params, solver_params)
			
			sanity_check_err = sanity_check_err + fs[1]
			if n % num_sanity_check == 0 then
				print(string.format("nEpoch %d; %d/%d; err = %f ", epoch, n, nBatches, sanity_check_err/num_sanity_check))
				sanity_check_err = 0
			end
		end

		print(string.format("nEpoch %d ; NLL train err = %f ", epoch, epoch_err/(nBatches)))
		local val_err = 0
		for n = 1, nBatches_val do
			local words, fc7, conv4, targets = torch.LongTensor(),torch.LongTensor(),torch.LongTensor(), torch.LongTensor() 
			words, fc7, conv4, targets = words:cuda(), fc7:cuda(), conv4:cuda(), targets:cuda() 
			local inputs = Utils.getNextBatch(ds_val, ds_val.indices, n, words, fc7, conv4, targets)
			fc7 = inputs[2]
			conv4 = inputs[3]
			local batchSize = words:size(1)
			local totalInput = {fc7, words, conv4}
			-- forward step
			local totalOutput, err = Train.foward(clones, protos, totalInput, targets, batchSize)
			val_err = val_err + err
		end
		print(string.format("nEpoch %d ; NLL val err = %f ", epoch, val_err/(nBatches_val)))

		if epoch % 1 == 0 then
			local filename = paths.concat(opt.model, 'nEpoch_' .. epoch .. os.date("_%m_%d_%Y_%H_%M_%S") .. '.net')
			torch.save(filename, protos)
		end
	end
end

return Train
