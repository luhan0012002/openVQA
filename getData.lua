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

function getData.getBatchFc7(ds, indices, start_idx, end_idx)
    local fc7_path = '../fc7'
    local img_id = torch.LongTensor()
    img_id = img_id:cuda()
    img_id:index(ds.img_id, 1, indices:sub(start_idx, end_idx))
    local fc7 = {}
    for i = 1,img_id:size(1) do
        feature_path = fc7_path..'/v7w_'..tostring(img_id[i])..'.jpg.t7'
        feature = torch.load(feature_path)
        feature:resize(feature:size(1),1)
        table.insert(fc7,feature:t())
    end
    local net = nn.JoinTable(1)
    net = net:cuda()
    return net:forward(fc7)
end

function getData.getBatchConv4(ds, indices, start_idx, end_idx)               
    local conv_path = '../conv'                           
    local img_id = torch.LongTensor()                                              
    img_id = img_id:cuda()                                                         
    img_id:index(ds.img_id, 1, indices:sub(start_idx, end_idx))                    
    local conv4 = {}                                                               
    for i = 1,img_id:size(1) do                                                    
        feature_path = conv_path..'/v7w_'..tostring(img_id[i])..'.jpg.t7'          
        feature = torch.load(feature_path)                                         
        feature = torch.permute(feature, 2, 3, 1)                               
        --feature = feature:type('torch.FloatTensor')                            
        --tmp = feature[{1,14,{}}] for checking correctly reshape                  
        reshapeNet = nn.Sequential()                                               
        :add(nn.Reshape(196, 512))                                              
        :add(nn.Reshape(1,196,512))                                             
        reshapeNet = reshapeNet:cuda()
        feature = reshapeNet:forward(feature:cuda())
        --feature = torch.expand(feature, rho, 196, 512)                           
        --feature = nn.Reshape(1,rho,196,512):forward(feature)                     
        --print(torch.all(feature[{1,1,14,{}}]:eq(tmp))) for checking correctly reshape
        table.insert(conv4, feature)                                            
    end                                                                         
    conv4 = nn.JoinTable(1):cuda():forward(conv4)                                     
    return conv4                                                                
end 

function getData.read(split, rho)
    local data_path
    if split == 'train' then
        data_path = '../data/dataset_v7w_telling_tokenized_train.json'    
        print('loading training data...')
    elseif split == 'val' then
        data_path = '../data/dataset_v7w_telling_tokenized_val.json'    
        print('loading validation data...')
    elseif split == 'test' then
        data_path = '../data/dataset_v7w_telling_tokenized_test.json'    
        print('loading testing data...')
    else
        print("wrong split !!!")
    end
    print(data_path)
    local wtoi_path = '../data/word2idx.json'
    local itow_path = '../data/idx2word.json'
    local cjson = require "cjson"
    local cjson2 = cjson.new()
    local cjson_safe = require "cjson.safe"
    local f = io.open(data_path, "r")
    local text = f:read("*all")
    f:close()
    local f = io.open(wtoi_path, "r")
    local wtoi_text = f:read("*all")
    f:close()
    local f = io.open(itow_path, "r")
    local itow_text = f:read("*all")
    f:close()
    local tableJson = cjson.decode(text)
    local wtoi = cjson.decode(wtoi_text)
    local itow = cjson.decode(itow_text)
    local ds = {} 
    local input = {}
    local target = {}
    local img_id = {}
    local question = {}
    local choices = {}
    local qa_id = {}
    local max_len = 0
    local tmp
    for i, img in ipairs(tableJson['images']) do
        if img['split'] == split then
            for j, qa_pair in ipairs(img['qa_pairs']) do
                ques = {}
                for w in qa_pair['question']:gmatch("[^%s$]+")  do
                    table.insert(ques, wtoi[w])
                end
                for k, multiple_choice in ipairs(qa_pair['multiple_choices']) do
                    dat = copy(ques)
                    --dat = ques
                    for w in multiple_choice:gmatch("[^%s$]+")  do
                        if wtoi[w] == nil then
                            table.insert(dat, wtoi['UNK'])
                        else    
                            table.insert(dat, wtoi[w])
                        end
                        if #dat >= rho then
                            break
                        end
                    end
                    for i = #dat+1, rho do
                        table.insert(dat, 1, 0)
                    end
                    table.insert(input, dat)
                    table.insert(target, 1)
                    table.insert(img_id, tonumber(img["image_id"]))
                    tmp, _ = string.gsub(qa_pair['question'], "%s(%p)", "%1")
                    table.insert(question, tmp)
                    tmp, _ = string.gsub(multiple_choice, "%s(%p)", "%1")
                    table.insert(choices, tmp)
                    table.insert(qa_id, tonumber(qa_pair["qa_id"]))
                end
                dat = copy(ques)
                for w in qa_pair['answer']:gmatch("[^%s$]+")  do
                    if wtoi[w] == nil then
                        table.insert(dat, wtoi['UNK'])
                    else    
                        table.insert(dat, wtoi[w])
                    end
                    if #dat >= rho then
                        break
                    end
                    if #dat > max_len then
                        max_len = #dat
                    end
                end
                for i = #dat+1, rho do
                    table.insert(dat, 1, 0)
                end
                table.insert(input, dat)
                table.insert(target, 2)
                table.insert(img_id, tonumber(img["image_id"]))
                tmp, _ = string.gsub(qa_pair['question'], "%s(%p)", "%1")
                table.insert(question, tmp)
                tmp, _ = string.gsub(qa_pair['answer'], "%s(%p)", "%1")
                table.insert(choices, tmp)
                table.insert(qa_id, tonumber(qa_pair["qa_id"]))
            end
        end
    end
    ds.input =  torch.LongTensor(input)
    ds.target =  torch.LongTensor(target)
    ds.img_id = torch.LongTensor(img_id)
    ds.size = #target
    --if split == 'test' then 
    ds.qa_id = qa_id
    ds.question = question
    ds.choices = choices
    --end
    --print(tablesize(itow))
    return ds
end

return getData

