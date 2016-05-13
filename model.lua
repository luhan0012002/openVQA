require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'SoftMaxInf'
require 'PrintIdentity'
require 'FastLSTM_padding'

local LSTM = require 'LSTM'

local model = {}
function model.buildEncoder(attMethod)
    local Att

    if attMethod == 'ReLU' then
        Att = require 'AttReLU'
    elseif attMethod == 'noActivation' then
        Att = require 'Att'
    elseif attMethod == 'AttStanford' then
        Att = require 'AttStanford'
    else print('wrong attention model!')
    end

    local protos = {}
    protos.lstm = nn.MaskZero(LSTM.lstm(srcEmbSize, hiddenSize), 1):cuda()
    --protos.lstm:maskZero(1)
    --protos.attention = Att.attention(hiddenSize, projectSize, convFeatureSize, numConvFeature)
    protos.wordEmbed = nn.LookupTableMaskZero(nSrcWords, srcEmbSize):cuda()
  --protos.classify = nn.Sequential():add(nn.SelectTable(2)):add(nn.Linear(hiddenSize, nClass)):add(nn.LogSoftMax()):cuda()
  --protos.criterion = nn.ClassNLLCriterion():cuda()

  --protos.lstm, protos.wordEmbed, protos.imageEmbed, protos.classify, protos.criterion = protos.lstm:cuda(), protos.wordEmbed:cuda(), protos.imageEmbed:cuda(), protos.classify:cuda, protos.criterion:cuda()

  return protos
end

function model.buildDecoder()
    local Att
    Att = require 'AttGlobal'
    local protos = {}
    protos.lstm = nn.MaskZero(LSTM.lstm(tgtEmbSize, hiddenSize), 1):cuda()
    protos.attention = nn.MaskZero(Att.attention(hiddenSize, rho), 1):cuda()
    protos.wordEmbed = nn.LookupTableMaskZero(nTgtWords, tgtEmbSize):cuda()
    protos.sample = nn.MaskZero(nn.Sequential():add(nn.Linear(hiddenSize, nTgtWords)):add(nn.LogSoftMax()), 1):cuda()
    protos.criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1):cuda()
    --protos.criterion = nn.ClassNLLCriterion():cuda()
    return protos
end

return model
--function utils.buildModel()
--[[
  lstm = FastLSTM_padding(hiddenSize, hiddenSize, rho)
  lstm:maskZero(1)

  rnn = nn.Sequential()
     :add(nn.ParallelTable()
         :add(nn.LookupTableMaskZero(nIndex, hiddenSize))
         :add(nn.Sequential()
              :add(nn.Linear(fcSize, hiddenSize))
              :add(nn.Reshape(1, hiddenSize, true))
             )  
         )
     :add(nn.JoinTable(2))
     :add(nn.SplitTable(1,2))
     :add(nn.Sequencer(lstm))
     :add(nn.SelectTable(-1)) -- this selects the last time-step of the rnn output sequence
     :add(nn.Linear(hiddenSize, nClass))
     :add(nn.LogSoftMax())

  rnn:cuda()
--]]
  --return {rnn}
--end
