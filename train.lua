require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'


local cjson = require 'cjson'
local model_utils = require 'model_utils'
local model = require 'model'
local Utils = require 'utils'
local Train = {}

--for tgt language 
local itow_path = './vocab/idx2tgt.json'
local f = io.open(itow_path, "r")
    local itow_tgt = f:read("*all")
    f:close()
local itow = cjson.decode(itow_tgt)

local wtoi_path = './vocab/tgt2idx.json'
local f = io.open(wtoi_path, "r")
    local wtoi_tgt = f:read("*all")
    f:close()
local wtoi = cjson.decode(wtoi_tgt)

function tablelength(T)
    local count = 0
    for _ in pairs(T) do count = count + 1 end
    return count
end

function Train.sample_single_word(distribution)
    --local idx = torch.squeeze(torch.multinomial(distribution, 1))
    local val, idx = torch.max(distribution, 2)
    --print(torch.squeeze(idx))
    word = itow[tostring(torch.squeeze(idx))]
    --word = itow[tostring(idx)]
    --[[
    local word
    if idx > 0 then
        assert(idx-2 >= 1 and idx-2 <= nTgtWords)
        word = itow[tostring(idx-2)]
    elseif idx == 2 then
        word = '</s>'
    else
        print('There is bug in sampling...')
    end
    ]]--
    return word
end

function Train.sample_seq(output_d, idx)
    local seqLen = #output_d
    local seq = ''
    for i = 1, seqLen do
        --print(torch.exp(output_d[i]:narrow(1, idx, 1)))
        local word = Train.sample_single_word(output_d[i]:narrow(1, idx, 1))
        if word == '</s>' then
            break
        end
        seq = seq..' '..word
    end
    return seq
    end

function Train.encode_forward(clones, protos, inputs, batchSize, encoderOutputs)
    -- inputs: {src_words, tgt_words}
    ht_e = {}
    ct_e = {}
    wordEmbed_e = {}
    table.insert(wordEmbed_e, clones['encoder']['wordEmbed'][1]:forward(inputs[1]:select(2,1)))
    local tmp = clones['encoder']['lstm'][1]:forward({wordEmbed_e[1], torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0)})
    table.insert(ht_e, tmp[1])
    table.insert(ct_e, tmp[2])

    for t = 2, rho do
        table.insert(wordEmbed_e, clones['encoder']['wordEmbed'][t]:forward(inputs[1]:select(2,t)))
        local tmp = clones['encoder']['lstm'][t]:forward({wordEmbed_e[t], ht_e[t-1], ct_e[t-1]})
        table.insert(ht_e, tmp[1])
        table.insert(ct_e, tmp[2])
    end 
    -- outputs: {ht, ct, wordEmbed}
    table.insert(encoderOutputs, ht_e)
    table.insert(encoderOutputs, ct_e)
    --table.insert(encoderOutputs, rt_e)
    table.insert(encoderOutputs, wordEmbed_e)
    end

function Train.decode_forward(clones, inputs, batchSize, encoderOutputs, decoderOutputs)
    -- inputs: {src_words, tgt_words, src_masks, tgt_masks}
    -- encoderOutputs = {ht_e, ct_e, wordEmbed_e}
    ht_d = {}
    ct_d = {}
    ht_hat_d = {}
    output_d = {}
    wordEmbed_d = {}
    loss = {}
    err = 0
    table.insert(wordEmbed_d, clones['decoder']['wordEmbed'][1]:forward(inputs[2]:select(2,1)))
    local tmp = clones['decoder']['lstm'][1]:forward({wordEmbed_d[1], encoderOutputs[1][rho], torch.CudaTensor(batchSize, hiddenSize):fill(0)})
    table.insert(ht_d, tmp[1])
    table.insert(ct_d, tmp[2])
    table.insert(ht_hat_d, clones['decoder']['attention'][1]:forward({ht_d[1], encoderOutputs[1]}))
    table.insert(output_d, clones['decoder']['sample'][1]:forward(ht_hat_d[1]))
    local l = clones['decoder']['criterion'][1]:forward(output_d[1], inputs[3]:select(2,1))
    table.insert(loss, l)
    err = err + loss[1]
    for t = 2, rho do
        table.insert(wordEmbed_d, clones['decoder']['wordEmbed'][t]:forward(inputs[2]:select(2,t)))
        --table.insert(rt, clones['attention'][t]:forward({ht[t-1], conv4}))
        local tmp = clones['decoder']['lstm'][t]:forward({wordEmbed_d[t], ht_d[t-1], ct_d[t-1]})
        table.insert(ht_d, tmp[1])
        table.insert(ct_d, tmp[2])
        table.insert(ht_hat_d, clones['decoder']['attention'][t]:forward({ht_d[t], encoderOutputs[1]}))
        table.insert(output_d, clones['decoder']['sample'][t]:forward(ht_hat_d[t]))
        table.insert(loss, clones['decoder']['criterion'][t]:forward(output_d[t], inputs[3]:select(2,t)))
        err = err + loss[t]
    end 
    table.insert(decoderOutputs, ht_d)
    table.insert(decoderOutputs, ct_d)
    table.insert(decoderOutputs,wordEmbed_d) 
    table.insert(decoderOutputs,loss) 
    table.insert(decoderOutputs,output_d)
    table.insert(decoderOutputs,ht_hat_d)
    return err
end 

function Train.encode_backward(clones, protos, inputs, enc_outputs, decoderGrad, batchSize)
    -- inputs: {src_words, tgt_words}
    -- enc_outputs {ht_e, ct_e, wordEmbed_e}
    -- prevGradInput {grad_word_emb, grad_ht, grad_ct}
    --print(prevGradInput[2][1])
   -- print(decoderGrad[2][1])
    local prevGradInput = clones['encoder']['lstm'][rho]:backward({enc_outputs[3][rho], enc_outputs[1][rho-1], enc_outputs[2][rho-1]}, {decoderGrad[1][2]+decoderGrad[2][rho], torch.CudaTensor(batchSize, hiddenSize):fill(0)})
    --clones['encoder']['attention'][t]:backward({enc_outputs[1][t-1], inputs[4]}, prevGradInput[4])
    clones['encoder']['wordEmbed'][rho]:backward(inputs[1]:select(2, rho), prevGradInput[1])
    for t = rho-1, 2, -1 do
        prevGradInput = clones['encoder']['lstm'][t]:backward({enc_outputs[3][t], enc_outputs[1][t-1], enc_outputs[2][t-1]}, {prevGradInput[2]+decoderGrad[2][t], prevGradInput[3]})
        --clones['encoder']['attention'][t]:backward({enc_outputs[1][t-1], inputs[4]}, prevGradInput[4])
        clones['encoder']['wordEmbed'][t]:backward(inputs[1]:select(2, t), prevGradInput[1])
    end
    prevGradInput = clones['encoder']['lstm'][1]:backward({enc_outputs[3][1], torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0)}, {prevGradInput[2]+decoderGrad[2][1], prevGradInput[3]})
    clones['encoder']['wordEmbed'][1]:backward(inputs[1]:select(2, 1), prevGradInput[1])
    --clones['encoder']['attention'][1]:backward({torch.CudaTensor(batchSize, hiddenSize):fill(0), conv4}, prevGradInput[4])
end

function Train.decode_backward(clones, inputs, enc_outputs, dec_outputs, batchSize)
    -- inputs: {q_words, a_words, fc7, conv4, targets}
    -- enc_outputs {ht_e, ct_e, wordEmbed_e}
    -- dec_outputs {ht_d, ct_d, wordEmbed_d, loss, output_d, ht_hat_d}
    local prevGradInput_lstm = {torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0), torch.CudaTensor(batchSize, hiddenSize):fill(0)}
    local decoderGrad = {}
    local encoderhGradInput = {}
    for i = 1, rho do
        table.insert(encoderhGradInput, torch.CudaTensor(batchSize, hiddenSize):fill(0))
    end
    for t = rho, 2, -1 do
        local gradOutput = clones['decoder']['criterion'][t]:backward(dec_outputs[5][t], inputs[3]:select(2,t))
        local prevGradInput = clones['decoder']['sample'][t]:backward(dec_outputs[6][t], gradOutput)
        local prevGradInput_criterion = clones['decoder']['attention'][t]:backward({dec_outputs[1][t], enc_outputs[1]}, prevGradInput)
        local sumPrevGradInput = prevGradInput_criterion[1] + prevGradInput_lstm[2] 
        for i = 1, rho do
            encoderhGradInput[i] = encoderhGradInput[i] + prevGradInput_criterion[2][i]
        end
        --print(prevGradInput_criterion[2])
        --print(dec_outputs[4][t]) 
        --print(prevGradInput)	
        --print(dec_outputs[1][t])	
        prevGradInput_lstm = clones['decoder']['lstm'][t]:backward({dec_outputs[3][t], dec_outputs[1][t-1], dec_outputs[2][t-1]}, {sumPrevGradInput, torch.CudaTensor(batchSize, hiddenSize):fill(0)})
        --print(inputs[2]:select(2,t))	
        clones['decoder']['wordEmbed'][t]:backward(inputs[2]:select(2,t), prevGradInput_lstm[1])
    end
    local gradOutput = clones['decoder']['criterion'][1]:backward(dec_outputs[5][1], inputs[3]:select(2,1))
    local prevGradInput = clones['decoder']['sample'][1]:backward(dec_outputs[6][1], gradOutput)	
    local prevGradInput_criterion = clones['decoder']['attention'][1]:backward({dec_outputs[1][1], enc_outputs[1]}, prevGradInput)
    local sumPrevGradInput = prevGradInput_criterion[1] + prevGradInput_lstm[2] 
    for i = 1, rho do
        encoderhGradInput[i] = encoderhGradInput[i] + prevGradInput_criterion[2][i]
    end
    local prevGradInput = clones['decoder']['lstm'][1]:backward({dec_outputs[3][1], enc_outputs[1][rho], torch.CudaTensor(batchSize, hiddenSize):fill(0)}, {sumPrevGradInput, torch.CudaTensor(batchSize, hiddenSize):fill(0)})
    clones['decoder']['wordEmbed'][1]:backward(inputs[2]:select(2,1), prevGradInput[1])
    table.insert(decoderGrad, prevGradInput)
    table.insert(decoderGrad, encoderhGradInput)
    

    return decoderGrad

end

function Train.generate_foward(clones, protos, inputs, outputs, batchSize, idx)
    outputs['encoderOutputs'] = {}
    outputs['decoderOutputs'] = {}
    input = {}
    table.insert(input, inputs[1][idx]:view(-1,rho))
    table.insert(input, inputs[2][idx]:view(-1,rho))
    Train.encode_forward(clones, protos, input, 1, outputs['encoderOutputs'])
    local seq = ""
    local ht_1 = outputs['encoderOutputs'][1][rho]
    --print(ht_1)
    local ct_1 = torch.CudaTensor(1, hiddenSize):fill(0)
    local word = '<s>'
    local count = 0
    local ref = ""
    local src = ""
    while word ~= '.' and word ~= '?' do
        local wordEmbed = clones['decoder']['wordEmbed'][count+1]:forward(torch.CudaTensor(1):fill(wtoi[word]))
        local tmp = clones['decoder']['lstm'][count+1]:forward({wordEmbed, ht_1, ct_1})
        ht_1 = tmp[1]
        ct_1 = tmp[2]
        local ht_hat = clones['decoder']['attention'][count+1]:forward({ht_1, outputs['encoderOutputs'][1]})
        local output_d = clones['decoder']['sample'][count+1]:forward(ht_hat)
        word = Train.sample_single_word(torch.exp(output_d))
        seq = seq ..' '.. word 
        count = count + 1
        if count >= rho then
            break
        end
    end
    for i = 1, rho do
        if inputs[2][idx][i] ~= 0 then
            ref = ref ..' '.. itow[tostring(inputs[2][idx][i])]
        end
    end
    local results = {}
    table.insert(results, ref)
    table.insert(results, seq)
    return results
end

function Train.foward(clones, protos, inputs, outputs, batchSize)
    outputs['encoderOutputs'] = {}
    outputs['decoderOutputs'] = {}
    Train.encode_forward(clones, protos, inputs, batchSize, outputs['encoderOutputs'])
    err = Train.decode_forward(clones, inputs, batchSize, outputs['encoderOutputs'], outputs['decoderOutputs'])
    --print(outputs['encoderOutputs'][1][1])
    --print(outputs['decoderOutputs'][4])
    ----print(outputs['decoderOutputs'][1][rho-1])
    --print(outputs['decoderOutputs'][6][1])
    --print("--------------------")
    return err 
end

function Train.backward(clones, protos, inputs, outputs, batchSize)
    --outputs of the forward function will be stored here
    --outputs.encoder {ht_e, ct_e, wordEmbed_e}
    --outputs.decoder {ht_d, ct_d, wordEmbed_d, loss, outputWordIdx}
    decoderGrad = Train.decode_backward(clones, inputs, outputs['encoderOutputs'], outputs['decoderOutputs'], batchSize)
    Train.encode_backward(clones, protos, inputs, outputs['encoderOutputs'], decoderGrad, batchSize)
end

function Train.test(protos, ds_test, ans_path)
    local f = io.open(ans_path, "w")
    clones = {}
    clones.encoder = {}
    clones.decoder = {}
    clones['encoder']['lstm'] = model_utils.clone_many_times(protos.encoder.lstm, rho, not protos.encoder.lstm.parameters)
    clones['encoder']['wordEmbed'] = model_utils.clone_many_times(protos.encoder.wordEmbed, rho, not protos.encoder.wordEmbed.parameters) -- 1 unit less than lstm
    --clones['encoder']['attention'] = model_utils.clone_many_times(protos.encoder.attention, rho+1, not protos.encoder.attention.parameters) 
    clones['decoder']['lstm'] = model_utils.clone_many_times(protos.decoder.lstm, rho, not protos.decoder.lstm.parameters)
    clones['decoder']['attention'] = model_utils.clone_many_times(protos.decoder.attention, rho, not protos.decoder.attention.parameters)
    clones['decoder']['wordEmbed'] = model_utils.clone_many_times(protos.decoder.wordEmbed, rho, not protos.decoder.wordEmbed.parameters) -- 1 unit less than lstm
    clones['decoder']['sample'] = model_utils.clone_many_times(protos.decoder.sample, rho, not protos.decoder.sample.parameters) -- 1 unit less than lstm
    clones['decoder']['criterion'] = model_utils.clone_many_times(protos.decoder.criterion, rho, not protos.decoder.criterion.parameters) -- 1 unit less than lstm
    nBatches_test = math.ceil(ds_test.size/batchSize)
    for n = 1, nBatches_test do
        local inputs = {}
        local outputs = {}
        Utils.getNextBatch(ds_test, n, inputs, 1)
        local batchSize = inputs[1]:size(1)
        
        for idx = 1, batchSize do
        -- forward step
            local results = Train.generate_foward(clones, protos, inputs, outputs, 1, idx)
            print(string.format("send id: %s ref %s ", idx, results[1]))
            print(string.format("send id: %s tgt %s ", idx, results[2]))
            f:write(string.format("<s>%s </s>\n", results[2]))
        end
    end
    f:close()
end

function Train.train_sgd(protos, ds, ds_val, solver_params)
    local nBatches_val = math.ceil(ds_val.size/batchSize)
    local nBatches = math.ceil(ds_train.size/batchSize)
    -- local lstm, wordEmbed, imageEmbed, classify, criterion = model[0],model[1],model[2],model[3],model[4]
    local params, grad_params = model_utils.combine_all_parameters(protos.encoder.lstm, protos.encoder.wordEmbed, protos.decoder.lstm, protos.decoder.wordEmbed, protos.decoder.sample)
    print('start cloning...')
    --clone model so their weight is shared
    clones = {}
    clones.encoder = {}
    clones.decoder = {}
    clones['encoder']['lstm'] = model_utils.clone_many_times(protos.encoder.lstm, rho, not protos.encoder.lstm.parameters)
    clones['encoder']['wordEmbed'] = model_utils.clone_many_times(protos.encoder.wordEmbed, rho, not protos.encoder.wordEmbed.parameters) -- 1 unit less than lstm
    --clones['encoder']['attention'] = model_utils.clone_many_times(protos.encoder.attention, rho+1, not protos.encoder.attention.parameters) 
    clones['decoder']['lstm'] = model_utils.clone_many_times(protos.decoder.lstm, rho, not protos.decoder.lstm.parameters)
    clones['decoder']['attention'] = model_utils.clone_many_times(protos.decoder.attention, rho, not protos.decoder.attention.parameters)
    clones['decoder']['wordEmbed'] = model_utils.clone_many_times(protos.decoder.wordEmbed, rho, not protos.decoder.wordEmbed.parameters) -- 1 unit less than lstm
    clones['decoder']['sample'] = model_utils.clone_many_times(protos.decoder.sample, rho, not protos.decoder.sample.parameters) -- 1 unit less than lstm
    clones['decoder']['criterion'] = model_utils.clone_many_times(protos.decoder.criterion, rho, not protos.decoder.criterion.parameters) -- 1 unit less than lstm
    print('done cloning!')

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
                local inputs = {} -- {q_words, a_words, fc7, conv4, targets}
                --outputs of the forward function will be stored here
                --outputs.encoder {ht_e, ct_e, rt_e, wordEmbed_e, imageEmbed_e}
                --outputs.decoder {ht_d, ct_d, wordEmbed_d, loss, outputWordIdx}
                local outputs = {} 

                --print('start getting next batch...')
                Utils.getNextBatch(ds, n, inputs)
                --print('done getting next batch!')
                local batchSize = inputs[1]:size(1)
                -- forward step  
                --print('start forwarding...')
                --if n==1 then
                --    print(Train.generate_foward(clones, protos, inputs, outputs, 1, 1))
                --end
                local err = Train.foward(clones, protos, inputs, outputs, batchSize)
                epoch_err = epoch_err + err
                --print('done forwarding!')		
                -- backward step
                --print('start backwarding...')
                Train.backward(clones, protos, inputs, outputs, batchSize)
                --print('done backwarding!')
                grad_params:clamp(-5, 5)

                return err, grad_params
            end
            local _, fs = optim.rmsprop(feval, params, solver_params)
            sanity_check_err = sanity_check_err + fs[1]
            if n % num_sanity_check == 0 then
                print(string.format("nEpoch %d; %d/%d; err = %f ", epoch, n, nBatches, sanity_check_err/num_sanity_check))
                sanity_check_err = 0
            end
    end

    print(string.format("nEpoch %d ; NLL train err = %f ", epoch, epoch_err/(nBatches)))
    local val_err = 0
    for n = 1, 1 do --nBatches_val do
        local inputs = {}
        local outputs = {}
        Utils.getNextBatch(ds_val, n, inputs)
        local batchSize = inputs[1]:size(1)
        
        for idx = 1, 10 do
        -- forward step
            local results = Train.generate_foward(clones, protos, inputs, outputs, 1, idx)
            print(string.format("send id: %s ref %s ", idx, results[1]))
            print(string.format("send id: %s tgt %s ", idx, results[2]))

        end
    end
    print(string.format("nEpoch %d ; NLL val err = %f ", epoch, val_err/(nBatches_val)))

    if epoch % 1 == 0 then
    local filename = paths.concat(opt.model, 'nEpoch_' .. epoch .. os.date("_%m_%d_%Y_%H_%M_%S") .. '.net')
    torch.save(filename, protos)
    end
    end
    end

    return Train
