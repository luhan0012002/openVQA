require 'io'
require 'nn'
require 'cunn'
require 'cutorch'

getData = {}

local function tablesize(t)
    local count = 0
    for key, value in pairs(t) do
        count = count + 1
    end
    return count
end

local function copy(original)
    local copy = {}
    for key, value in pairs(original) do
        copy[key] = value
    end
    return copy
end


function getData.read(split, src, rho)
    local sent_path = './data/' .. split .. '.' .. src
    local wtoi_path = './vocab/' .. src .. '2idx.json' 
    local itow_path = './vocab/idx2' .. src .. '.json' 
   
    print(wtoi_path)
    --setting up json
    local cjson = require "cjson"
    local cjson2 = cjson.new()
    local cjson_safe = require "cjson.safe"
    
    local f = io.open(sent_path, "r")
    local corpus = f:read("*all")
    f:close()

    local f = io.open(wtoi_path, "r")
    local wtoi_text = f:read("*all")
    f:close()

    local f = io.open(itow_path, "r")
    local itow_text = f:read("*all")
    f:close()

    local wtoi = cjson.decode(wtoi_text)
    local itow = cjson.decode(itow_text)
    
    local sents = {}
    local sents_next = {}
    local masks = {}
    for s in corpus:gmatch("[^\n]+") do
        local sent = {}
        local mask = {}
        local sent_next = {}
        for w in s:gmatch("[^%s$]+")  do
            if wtoi[w] == nil then
                table.insert(sent, wtoi['<unk>'])
                table.insert(mask, 1)
                if #sent > 1 then
                    table.insert(sent_next, wtoi['<unk>'])
                end
            elseif wtoi[w] == '</s>' then
                table.insert(sent, 0)
                table.insert(mask, 0)
                if #sent > 1 then
                    table.insert(sent_next, 1)
                end
            else    
                table.insert(sent, wtoi[w])
                table.insert(mask, 1)
                if #sent > 1 then
                    table.insert(sent_next, wtoi[w])
                end
            end
            if #sent >= rho then
                break
            end
        end

        --padding
        for i = #sent+1, rho do
            if src == 'tgt' then
                --pad 1 for tgt, it won't effect the results (just not 0)
                table.insert(sent, 0)
                table.insert(mask, 0)
                table.insert(sent_next, 1)
            else
                table.insert(sent, 1 ,0)
                table.insert(mask, 1, 0)
                table.insert(sent_next, 1, 1)
            end
        end
        table.insert(sent_next, 1)
        table.insert(sents, sent)
        table.insert(sents_next, sent_next)
        table.insert(masks, mask)
    end
    local input = {}
    sents = torch.DoubleTensor(sents)
    sents_next = torch.DoubleTensor(sents_next)
    masks = torch.DoubleTensor(masks)
    input.sents = sents
    input.sents_next = sents_next
    input.masks = masks
    

    return input
end

return getData

