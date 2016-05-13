require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'SoftMaxInf'
require 'PrintIdentity'


local model = require 'model'
local Train = require 'train'
local Utils = require 'utils'
local GenerateAns = require 'generateAns'
-- hyper-parameters 
batchSize = 128
rho = 15 -- sequence length
hiddenSize = 512
srcEmbSize = 128
tgtEmbSize = 128



nSrcWords = 3436--5305 -- input words
nTgtWords = 3121

nEpoch = 50
num_sanity_check = 100

local sgd_params = {
    learningRate = 1e-3,
    weightDecay = 1e-4
}

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-train', 1, '1 or 0')
cmd:option('-pretrained_model', '', 'pretrained model path')
cmd:option('-model', './model/', 'output model path')
cmd:option('-ans_path', '', 'ans path')
cmd:option('-Att','ReLU', 'ReLU, noActivation, or AttStanford')
cmd:option('-gpu', '1', 'gpu device id')
opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)


-- load model --
local protos = {}
if path.exists(opt.pretrained_model) then
    print(string.format("Loading model from %s...", opt.pretrained_model))
    protos = torch.load(opt.pretrained_model)
else
    print(string.format("Initializing model..."))
    local encoder = model.buildEncoder(opt.Att)
    local decoder = model.buildDecoder()

    for k, v in pairs(encoder) do
        v = v:cuda()
    end
    for k, v in pairs(decoder) do 
        v = v:cuda()
    end

    protos.encoder = encoder
    protos.decoder = decoder
end


if opt.train == 1 then
	print('Training...')
	ds_train = Utils.loadData('train', true)
        ds_val = Utils.loadData('dev', false)
	Train.train_sgd(protos, ds_train, ds_val, sgd_params)
else
	print('Testing...')
	ds_test = Utils.loadData('test', false)
	Train.test(protos, ds_test, opt.ans_path)
end


