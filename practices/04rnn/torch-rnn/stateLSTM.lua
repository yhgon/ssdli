local stateLSTM, parent = torch.class('stateLSTM', 'cudnn.LSTM')

function stateLSTM:resetStates()
   if self.hiddenInput then 
      self.hiddenInput:zero()
   end
   if self.cellInput then
      self.cellInput:zero()
   end
end


function stateLSTM:updateOutput(input)
   if (self.hiddenInput and input:size(2) ~= self.hiddenInput:size(2)) then
      self.hiddenInput = nil
   end
   if (self.cellInput and input:size(2) ~= self.cellInput:size(2)) then
      self.cellInput = nil
   end  
   
   local output = parent.updateOutput(self, input)
   self.hiddenInput = self.hiddenOutput:clone()
   self.cellInput = self.cellOutput:clone()
   return output
end