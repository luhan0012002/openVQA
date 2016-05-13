local SoftMaxInf, _ = torch.class('nn.SoftMaxInf', 'nn.Module')

function SoftMaxInf:updateOutput(input)
   input[input:eq(0)]=-math.huge
   input.THNN.SoftMax_updateOutput(
      input:cdata(),
      self.output:cdata()
   )
   return self.output
end

function SoftMaxInf:updateGradInput(input, gradOutput)
   input[input:eq(0)]=-math.huge
   input.THNN.SoftMax_updateGradInput(
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.output:cdata()
   )
   self.gradInput[self.gradInput:ne(self.gradInput)] = 0
   return self.gradInput
end
