require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

local model = require 'model'
local Train = require 'train'
local Utils = require 'utils'
local GenerateAns = require 'generateAns'

-- hyper-parameters 
batchSize = 64
rho = 34 -- sequence length
hiddenSize = 512
projectSize = 1
convFeatureSize = 512
numConvFeature = 196
nIndex = 5305 -- input words
nClass = 2 -- output classes
nEpoch = 100
fcSize = 4096
num_sanity_check = 300

local sgd_params = {
	learningRate = 1e-2
}

cmd = torch.CmdLine()
cmd:text('Options')
cmd:option('-train', 1, '1 or 0')
cmd:option('-pretrained_model', '', 'pretrained model path')
cmd:option('-model', './', 'output model path')
cmd:option('-ans_path', '', 'ans path')
cmd:option('-Att','ReLU', 'ReLU, noActivation, or Stanford')
cmd:option('-gpu', '1', 'gpu device id')
opt = cmd:parse(arg)
cutorch.setDevice(opt.gpu)
-- load model --
local protos
if path.exists(opt.pretrained_model) then
    print(string.format("Loading model from %s...", opt.pretrained_model))
    protos = torch.load(opt.pretrained_model)
else
    print(string.format("Initializing model..."))
    protos = model.buildModel(opt.Att)
    protos.lstm = protos.lstm:cuda()
    protos.wordEmbed = protos.wordEmbed:cuda()
    protos.imageEmbed = protos.imageEmbed:cuda()
    protos.classify = protos.classify:cuda()
    protos.attention = protos.attention:cuda()
end

if opt.train == 1 then
	print('Training...')
	ds_train = Utils.loadData('train', true)
	ds_val = Utils.loadData('val', false)
	Train.train_sgd(protos, ds_train, ds_val, sgd_params)
else
	print('Testing...')
	ds_test = Utils.loadData('test', false)
	GenerateAns.generateAns(ds_test, protos, opt.ans_path)
end


