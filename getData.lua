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
    local img_id = torch.Tensor()
    --img_id = img_id:cuda()
    img_id:index(ds.img_id, 1, indices:sub(start_idx, end_idx))
    local fc7 = {}
    for i = 1,img_id:size(1) do
        feature_path = fc7_path..'/v7w_'..tostring(img_id[i])..'.jpg.t7'
        feature = torch.load(feature_path)
        feature:resize(feature:size(1),1)
        table.insert(fc7,feature:t())
    end
    local net = nn.JoinTable(1)
    --net = net:cuda()
    return net:forward(fc7)
end

function getData.getBatchConv4(ds, indices, start_idx, end_idx)               
    local conv_path = '../conv'                           
    local img_id = torch.Tensor()                                              
    --img_id = img_id:cuda()                                                         
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
    return nn.JoinTable(1):forward(conv4)                                                                
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
    --including source(question) and target(answer) 
    local input = {}
    local target = {}
    local img_id = {}
    local question = {}
    local answer = {}
    local qa_id = {}
    local max_len = 0
    local tmp
    --index 0 is for padding
    --index 1 is for <s>
    --index 2 in for </s>

    for i, img in ipairs(tableJson['images']) do
        if img['split'] == split then
            for j, qa_pair in ipairs(img['qa_pairs']) do
                --question
                ques = {}
                for w in qa_pair['question']:gmatch("[^%s$]+")  do
                    if wtoi[w] == nil then
                        table.insert(ques, wtoi['UNK'] + 2)
                    else    
                        table.insert(ques, wtoi[w] + 2)
                    end
                    -- cut of if question is longer than rho
                    if #ques >= rho then
                        break
                    end
                end

                --padding
                for i = #ques+1, rho do
                    table.insert(ques, 1, 0)
                end

                --answer
                ans = {}
                --target, different padding, pads 1 as dummy 
                tar = {}
                -- insert <s>
                table.insert(ans, 1)
                for w in qa_pair['answer']:gmatch("[^%s$]+")  do
                    if wtoi[w] == nil then
                        table.insert(tar, wtoi['UNK'] + 2)
                        table.insert(ans, wtoi['UNK'] + 2)
                    else    
                        table.insert(tar, wtoi[w] + 2)
                        table.insert(ans, wtoi[w] + 2)
                    end
                    -- cut of if answer is longer than rho
                    if #ans >= rho then
                        break
                    end
                end
                -- insert </s>
                table.insert(tar, 2)
                --padding
                for i = #ans+1, rho do
                    table.insert(ans, 1, 0)
                    --pad 1 as dummy
                    table.insert(tar, 1, 1) 
                end
                table.insert(input, ques)
                table.insert(target, ans)
                table.insert(img_id, tonumber(img["image_id"]))
                tmp, _ = string.gsub(qa_pair['question'], "%s(%p)", "%1")
                table.insert(question, tmp)
                tmp, _ = string.gsub(qa_pair['answer'], "%s(%p)", "%1")
                table.insert(answer, tmp)
                table.insert(qa_id, tonumber(qa_pair["qa_id"]))
            end
        end
    end
    ds.input =  torch.Tensor(input)
    ds.target =  torch.Tensor(target)
    ds.img_id = torch.Tensor(img_id)
    ds.size = #target
    --if split == 'test' then 
    ds.qa_id = qa_id
    ds.question = question
    ds.answer = answer
    --end
    --print(tablesize(itow))
    return ds
end

return getData

