local stateRNN, parent = torch.class('stateRNN', 'cudnn.RNNTanh')

function stateRNN:resetStates()
   if self.hiddenInput then 
      self.hiddenInput:zero()
   end
end


function stateRNN:updateOutput(input)
   if (self.hiddenInput and input:size(2) ~= self.hiddenInput:size(2)) then
      self.hiddenInput = nil
   end
   
   local output = parent.updateOutput(self, input)
   self.hiddenInput = self.hiddenOutput:clone()
   return output
end
