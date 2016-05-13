require 'nngraph'
require 'SoftMaxInf'
print(a)
require 'PrintIdentity'

local Att = {}
function Att.attention(rnn_size, num_src)
  --[[
  local inputs = {}
  table.insert(inputs, nn.Identity()())   -- h at time t-1
  table.insert(inputs, nn.Identity()())   -- C(I)
  local prev_h = inputs[1]
  local C = inputs[2] -- (batchSize, 196, 512)
  --]]
  --[[

  idea:

   join table
   repeat table
   multiply get a(score)
   multiply again -> rt

  
  --]] 
  local current_h = nn.Identity()()
  --local src_hs = PrintIdentity()()
  local src_hs = nn.JoinTable(2)()     
  local src_hs_1 = nn.View(-1,num_src, rnn_size):setNumInputDims(2)(src_hs)
 
  local tmp2 = nn.View(-1,1,rnn_size):setNumInputDims(2)(current_h)
  
  
  local score = nn.MM(false, true)({tmp2, src_hs_1})
  --local a = PrintIdentity()(nn.SoftMax()(nn.View(num_src):setNumInputDims(2)(score)))
  --local a = PrintIdentity()(nn.SoftMaxInf()(nn.View(num_src):setNumInputDims(2)(score)))
  --local a = nn.SoftMaxInf()(nn.View(num_src):setNumInputDims(2)(score))
  local a = nn.SoftMax()(nn.View(num_src):setNumInputDims(2)(score))
  --local preactivations = nn.CAddTable()({tmp1, tmp2})
  --local preactivations = nn.CAddTable()({C2h_reshape, h_repeat})
  --local tmp3 = PrintIdentity() (preactivations)
  --local preactivations_reshape = nn.View(project_size):setNumInputDims(1)(preactivations)
  --local activations = nn.Tanh()(preactivations_reshape)


  --local e = nn.Linear(project_size, 1)(activations)
  --local e_reshape = nn.View(1,num_conv_feature):setNumInputDims(2)(e)
  --local e_reshape = nn.Reshape(batchSize, num_conv_feature, 1,false)(e)
  --local a = nn.SoftMax()(e_reshape) 

  --local tmp4 = PrintIdentity()(a)
  --local r = PrintIdentity()(nn.MixtureTable()({a, src_hs_1}))
  local r = nn.MixtureTable()({a, src_hs_1})
  --local r = nn.MM(false, false)({a, src_hs_1})
  --local r_reshape = nn.Reshape(rnn_size)(r) 
  local r_linear = nn.Linear(rnn_size, rnn_size)(r)
  local h_linear = nn.Linear(rnn_size, rnn_size)(current_h)
  local sum_h = nn.CAddTable()({r_linear, h_linear})
  local h_hat = nn.Tanh()(sum_h)

  -- module outputs
  outputs = {}
  table.insert(outputs, h_hat)

  -- packs the graph into a convenient module with standard API (:forward(), :backward())
  return nn.gModule({current_h, src_hs}, outputs)
end

return Att
