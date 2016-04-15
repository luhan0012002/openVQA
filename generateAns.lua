require 'io'
require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'
local cjson = require "cjson"
local Train = require "train"
local Utils = require 'utils'
local model_utils = require 'model_utils'

local GenerateAns = {}


function GenerateAns.generateAns(ds_test, protos, ans_path)

    clones = {}
    clones['lstm'] = model_utils.clone_many_times(protos.lstm, rho+1, not protos.lstm.parameters)
    clones['wordEmbed'] = model_utils.clone_many_times(protos.wordEmbed, rho, not protos.wordEmbed.parameters) -- 1 unit less than lstm
    clones['attention'] = model_utils.clone_many_times(protos.attention, rho+1, not protos.attention.parameters)

    --nBatches_test = math.ceil(ds_test.size/batchSize)
    nBatches_test = math.ceil(ds_test.size/batchSize)
    


    local candidates = {}
    local answers = {}
    for n = 1, nBatches_test do
	print(string.format("%d/%d", n, nBatches_test))
        -- get next training/testing batch
        local words, fc7, conv4, targets = torch.LongTensor(),torch.LongTensor(),torch.LongTensor(), torch.LongTensor() 
        words, fc7, conv4, targets = words:cuda(), fc7:cuda(), conv4:cuda(), targets:cuda()

        -- get mini-batch
        local inputs, targets = Utils.getNextBatch(ds_test, ds_test.indices, n, words, fc7, conv4, targets)
        words = inputs[1]
        fc7 = inputs[2]
        conv4 = inputs[3]
        local local_batchSize = words:size(1)
	if local_batchSize < batchSize then
		assert(n == nBatches_test)
		words = torch.cat(words, torch.CudaTensor(batchSize-local_batchSize, rho):fill(0), 1)
		fc7 = torch.cat(fc7, torch.CudaTensor(batchSize-local_batchSize, fcSize):fill(0), 1)
		conv4 = torch.cat(conv4, torch.CudaTensor(batchSize-local_batchSize, 196, 512):fill(0), 1)
		targets = torch.cat(targets, torch.CudaTensor(batchSize-local_batchSize):fill(0), 1)
	end
        local totalInput = {fc7, words, conv4}
        -- forward step
	local batchSize = words:size(1)
        local totalOutput, err = Train.foward(clones, protos, totalInput, targets, batchSize)
	print(err)

        local outputs = totalOutput[6]
        local nQuestions = math.min(batchSize/4, local_batchSize/4 )
	local start_idx = (n-1) * batchSize+1
        local end_idx = n * batchSize
        if end_idx > ds_test.size then
                end_idx = ds_test.size
        end

        for q = 1, nQuestions do
            val, id = torch.max(outputs:sub((q - 1) * 4 + 1, q * 4), 1)
            local ans = {}
            local candidates = {}
            local a = {}
            a.answer = ds_test.choices[start_idx + (q - 1) * 4 + id[1][2] - 1]
            table.insert(candidates, a)
            ans.candidates = candidates
            ans.question = ds_test.question[start_idx + (q - 1) * 4 + id[1][2] - 1]
            ans.qa_id = ds_test.qa_id[start_idx + (q - 1) * 4  + id[1][2] - 1]
            table.insert(answers, ans)
        end
    end
    json_text = cjson.encode(answers)
    print(json_text)
    local f = io.open(ans_path, "w")
    f:write(json_text) 
    f:close()

end
return GenerateAns
