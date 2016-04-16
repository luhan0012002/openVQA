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

function Train.encode_forward(clones, protos, inputs, batchSize, encoderOutputs)
	-- inputs: {words, fc7, conv4, targets}
	ht_e = {}
	ct_e = {}
	rt_e = {}
	wordEmbed_e = {}
	imageEmbed_e = protos.encoder.imageEmbed:forward(inputs[2])
	table.insert(rt_e, clones['encoder']['attention'][1]:forward({torch.CudaTensor(batchSize, hiddenSize):fill(0), inputs[3]}))
	local tmp = clones['encoder']['lstm'][1]:forward({imageEmbed, torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0), rt_e[1]})
	table.insert(ht_e, tmp[1])
	table.insert(ct_e, tmp[2])
	for t = 2, rho+1 do
		table.insert(wordEmbed_e, clones['encoder']['wordEmbed'][t-1]:forward(inputs[1]:select(2,t-1)))
		table.insert(rt_e, clones['encoder']['attention'][t]:forward({ht[t-1], inputs[3]}))
		local tmp = clones['encoder']['lstm'][t]:forward({wordEmbed_e[t-1], ht_e[t-1], ct_e[t-1], rt_e[t]})
		table.insert(ht_e, tmp[1])
		table.insert(ct_e, tmp[2])
	end 
	encoderOutputs = {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
end

function Train.decode_forward(clones, inputs, batchSize, batchSize, decoderOutputs, encoderOutputs)
	-- inputs: {words, fc7, conv4, targets}
	-- encoderOutputs = {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
	ht_d = {}
	ct_d = {}
	output_d = {}
	lt_d = {}
	wordEmbed_d = {}
	table.insert(wordEmbed_d, clones['decoder']['wordEmbed'][1]:forward(inputs[4]:select(2,1)))
	local tmp = clones['decode']['lstm'][1]:forward({wordEmbed_e[1], torch.CudaTensor(batchSize, hiddenSize):fill(0), encoderOutputs[1][rho]})
	table.insert(ht_d, tmp[1])
	table.insert(ct_d, tmp[2])
	local err = 0
	for t = 2, rho do
		table.insert(wordEmbed_d, clones['decoder']['wordEmbed'][t]:forward(targets:select(2,t)))
		--table.insert(rt, clones['attention'][t]:forward({ht[t-1], conv4}))
		local tmp = clones['decode']['lstm'][t]:forward({wordEmbed[t], ht[t], ct[t]})
		table.insert(ht_d, tmp[1])
		table.insert(ct_d, tmp[2])
		table.insert(output_d, clones['decoder']['sample'][t]:forward({wordEmbed_d[t], ht_d[t], ct[t]})
		table.insert(loss, clones['decoder']['criterion']:forward(output_d[t], targets:select(2,t+1)))
		err = err + loss[t]
	end 
	decoderOutputs = {ht_d, ct_d, wordEmbed_d, loss, output_d}
	return err
end 

function Train.encode_backward(clones, inputs, enc_outputs)
	-- inputs: {words, fc7, conv4, targets}
	-- enc_outputs {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
	for t = rho, 2, -1 do
		prevGradInput = clones['lstm'][t]:backward({enc_outputs[4][t-1], enc_outputs[1][t-1], enc_outputs[2][t-1], enc_outputs[3][t]}, {prevGradInput[2], prevGradInput[3]})
		clones['encoder']['attention'][t]:backward({enc_outputs[1][t-1], inputs[3]}, prevGradInput[4])
		clones['encoder']['wordEmbed'][t-1]:backward(inputs[1]:select(2, t-1), prevGradInput[1])
	end
	prevGradInput = clones['lstm'][1]:backward({imageEmbed, torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0), rt[1]}, {prevGradInput[2], prevGradInput[3]})
	clones['attention'][1]:backward({torch.CudaTensor(batchSize, hiddenSize):fill(0), conv4}, prevGradInput[4])
	protos.imageEmbed:backward(fc7, prevGradInput[1])
end

function Train.decode_backward(clones, inputs, enc_outputs, dec_outputs, prevGradInput)
	-- inputs: {words, fc7, conv4, targets}
	-- enc_outputs {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
	-- dec_outputs {ht_d, ct_d, wordEmbed_d, loss, output_d}
	for t = rho, 2, -1 do	
		local prevGradInput = clones['lstm'][t]:backward({outputs[3][t], outputs[1][t], outputs[2][t]}, {prevGradInput[2], prevGradInput[3]})
		clones['decoder']['wordEmbed'][t]:backward(targets:select(2,t), prevGradInput)
	end
	local gradOutput = clones['decoder']['criterion']:backward(inputs[5][1], inputs[4][1])
	local prevGradInput = clones['decoder'][sample]:backward({outputs[3][1], outputs[1][1], outputs[2][1]}, gradOutput)	
	local prevGradInput = clones['lstm'][t]:backward({outputs[3][1], enc_outputs[3][rho], torch.CudaTensor(batchSize, hiddenSize):fill(0)}, {prevGradInput[2], prevGradInput[3]})
	clones['decoder']['wordEmbed'][t]:backward(targets:select(2,t), prevGradInput)

	return prevGradInput

end

function Train.foward(clones, inputs, outputs)
	outputs.encoderOutputs = {}
	outputs.decoderOutputs = {}
	Train.encode_forward(clones, inputs, batchSize, outputs.encoderOutputs)
	err = Train.decode_forward(clones, inputs, batchSize, forward_outputs.encoderOutputs, outputs.decoderOutputs)

end

function Train.backward(clones, inputs, outputs)
	--outputs of the forward function will be stored here
	--outputs.encoder {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
	--outputs.decoder {ht_d, ct_d, wordEmbed_d, loss, outputWordIdx}

	prevGrad = Train.decode_backward(clones, inputs, outputs.encoder, outputs.decoder)
	Train.encode_backward(clones, inputs, outputs.encoder, prevGrad)
end

function Train.train_sgd(ds, ds_val, solver_params)
	local nBatches_val = math.ceil(ds_val.size/batchSize)
	local nBatches = math.ceil(ds_train.size/batchSize)
	-- local lstm, wordEmbed, imageEmbed, classify, criterion = model[0],model[1],model[2],model[3],model[4]
	local params, grad_params = model_utils.combine_all_parameters(protos.encoder.lstm, protos.encoder.attention, protos.encoder.wordEmbed, protos.encoder.imageEmbed, protos.decoder.lstm, protos.decoder.sample)
	--clone model so their weight is shared
	clones = {}
	clones['encoder']['lstm'] = model_utils.clone_many_times(protos.encoder.lstm, rho+1, not protos.encoder.lstm.parameters)
	clones['encoder']['wordEmbed'] = model_utils.clone_many_times(protos.encoder.wordEmbed, rho, not protos.encoder.wordEmbed.parameters) -- 1 unit less than lstm
	clones['encoder']['attention'] = model_utils.clone_many_times(protos.encoder.attention, rho+1, not protos.encoder.attention.parameters) 
	clones['decoder']['lstm'] = model_utils.clone_many_times(proto.decoder.lstm, rho, not protos.decoder.lstm.parameters)
	clones['decoder']['wordEmbed'] = model_utils.clone_many_times(protos.encoder.wordEmbed, rho, not protos.encoder.wordEmbed.parameters) -- 1 unit less than lstm

	for epoch = 1, nEpoch do
		local epoch_err = 0
		local sanity_check_err = 0
		for n = 1, nBatches do
			-- feval function for sgd solver
			local function feval(x)
				if x ~= params then
					params:copy(x)
				end
				grad_params:zero()

				-- get mini-batch
				local inputs = {} -- {words, fc7, conv4, targets}
				--outputs of the forward function will be stored here
				--outputs.encoder {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
				--outputs.decoder {ht_d, ct_d, wordEmbed_d, loss, outputWordIdx}
				local outputs = {} 


				Utils.getNextBatch(ds, n, inputs)
				local batchSize = inputs[1]:size(1)
				-- forward step  
				local err = Train.foward(clones, protos, inputs, outputs, batchSize)
				epoch_err = epoch_err + err			
				-- backward step
				Train.backward(clones, protos, inputs, outputs)
				grad_params:clamp(-5, 5)

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
			local inputs = Utils.getNextBatch(ds_val, n, words, fc7, conv4, targets)
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
