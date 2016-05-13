require 'nn'
require 'rnn'
require 'getData'
require 'optim'
require 'cutorch'
require 'cunn'
require 'PrintIdentity'
require 'FastLSTM_padding'

Utils = {}

function Utils.getNextBatch(ds, n, inputs, test)
    local src_words, tgt_words, tgt_next_words, src_masks, tgt_masks = torch.CudaTensor(), torch.CudaTensor(), torch.CudaTensor(), torch.CudaTensor(), torch.CudaTensor()
    local start_idx = (n-1) * batchSize+1
    local end_idx = n * batchSize
    if end_idx > ds.size then
	end_idx = ds.size
    end
    -- inputs{src_words, tgt_words}  
    src_words:index(ds.src, 1, ds.indices:sub(start_idx, end_idx))
    table.insert(inputs, src_words)

    if test~=1 then
        tgt_words:index(ds.tgt, 1, ds.indices:sub(start_idx, end_idx))
        table.insert(inputs, tgt_words)
        tgt_next_words:index(ds.tgt_next, 1, ds.indices:sub(start_idx, end_idx))
        table.insert(inputs, tgt_next_words)
        tgt_masks:index(ds.tgt_mask, 1, ds.indices:sub(start_idx, end_idx))
        table.insert(inputs, tgt_masks)
    end
    
   
    
    src_masks:index(ds.src_mask, 1, ds.indices:sub(start_idx, end_idx))
    table.insert(inputs, src_masks)
end

function Utils.loadData(split, isShuffle)
    local ds = {}
    
    local src = getData.read(split, 'src', rho)
    ds.src = src.sents:cuda()
    ds.src_mask = src.masks:cuda()
    if split ~= 'test' then
        local tgt = getData.read(split, 'tgt', rho)
        ds.tgt = tgt.sents:cuda() 
        ds.tgt_next = tgt.sents_next:cuda()
        ds.tgt_mask = tgt.masks:cuda() 
    end
    
    ds.size = src.sents:size(1)

    local indices = torch.LongTensor(ds.size)
	
    if isShuffle then
	indices:randperm(ds.size)
    else
        for i = 1,indices:size(1) do
	    indices[i] = i 
        end
    end	

    ds.indices = indices:cuda()
    return ds
end


return Utils
