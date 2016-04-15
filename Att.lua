require 'nngraph'
require 'PrintIdentity'

local Att = {}
function Att.attention(rnn_size, project_size, conv_feature_size, num_conv_feature)
  --[[
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  table.insert(inputs, nn.Identity()())   -- C(I)
  local prev_h = inputs[1]
  local C = inputs[2] -- (batchSize, 196, 512)
  --]]
  
  local prev_h = nn.Identity()()
  local C = nn.Identity()()


  --local tmpp = PrintIdentity()(prev_h)
  --local tmp2 = PrintIdentity()(C)
  
  local C_reshape = nn.View(conv_feature_size):setNumInputDims(1)(C)
  
  --local C_reshape = nn.Reshape(196*batchSize,512)(C)  --reshape C (batchSize*196, 512)
  --local tmp = PrintIdentity()(C_reshape)
  
  local C2h = nn.Linear(conv_feature_size, project_size)(C_reshape)      -- C_reshape to hidden (batchSize*196, 512)
  --C2h = PrintIdentity()(C2h)
 
  local C2h_reshape = nn.View(num_conv_feature, project_size):setNumInputDims(2)(C2h)
  --local tmp1 = PrintIdentity()(C2h_reshape)


  local h2h = nn.Linear(rnn_size, project_size)(prev_h)           -- hidden to hidden (batchSize, 512)
  
  local h_repeat = nn.Replicate(num_conv_feature , 2, 2)(h2h)   -- repeat h (batchSize*196, 512)
  --local tmp2 = PrintIdentity()(h_repeat)
  --local h_repeat = nn.Replicate(num_conv_feature , 2, 2)(h2h)   -- repeat h (batchSize*196, 512)
  --local h_project = nn.View(project_size):setNumInputDims(1)(h_repeat)
  --local h_repeat = nn.Sequential():add(nn.Replicate(num_conv_feature , 2, 2)):add(nn.Reshape(batchSize*num_conv_feature, project_size, false))(h2h)   -- repeat h (batchSize*196, 512)
  --local h_repeat = nn.Sequential():add(nn.Replicate(num_conv_feature , 2, 2)):add(nn.Contiguous()):add(nn.View(project_size):setNumInputDims(1))(h2h)   -- repeat h (batchSize*196, 512)
  


  --local preactivations = nn.CAddTable()({tmp1, tmp2})
  local preactivations = nn.CAddTable()({C2h_reshape, h_repeat})
  --local tmp3 = PrintIdentity() (preactivations)
  local preactivations_reshape = nn.View(project_size):setNumInputDims(1)(preactivations)
  local activations = nn.Tanh()(preactivations_reshape)


  local e = nn.Linear(project_size, 1)(activations)
  local e_reshape = nn.View(1,num_conv_feature):setNumInputDims(2)(e)
  --local e_reshape = nn.Reshape(batchSize, num_conv_feature, 1,false)(e)
  local a = nn.SoftMax()(e_reshape) 

  --local tmp4 = PrintIdentity()(a)

  local r = nn.MM(false, false)({a, C})
  local r_reshape = nn.Reshape(conv_feature_size)(r) 

  -- module outputs
  outputs = {}
  table.insert(outputs, r_reshape)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule({prev_h, C}, outputs)
end

return Att
