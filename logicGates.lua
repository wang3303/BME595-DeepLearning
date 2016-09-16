k = require 'NeuralNetwork'
require 'torch'
local M = {}

function M.AND(x,y)
	i = (x == true) and 1 or 0
	j = (y == true) and 1 or 0
	
	input = torch.Tensor({i,j})
	k.build({2,1})
	lay = k.getLayer(1)
	lay[1]= torch.Tensor{-30,20,20}
	r = k.forward(input)
	return (r[1]>0.5) and true or false
end

function M.OR(x,y)
	i = (x == true) and 1 or 0
	j = (y == true) and 1 or 0
	input = torch.Tensor({i,j})
	k.build({2,1})
	lay = k.getLayer(1)
	lay[1]= torch.Tensor{-10,20,20}
	r = k.forward(input)
	return (r[1]>0.5) and true or false
end

function M.NOT(x)
	i = (x == true) and 1 or -1
	input = torch.Tensor({i})
	k.build({1,1})
	lay = k.getLayer(1)
	lay[1]= torch.Tensor{0,-10}
	r = k.forward(input)
	return (r[1]>0.5) and true or false
end

function M.XOR(x,y)
	i = (x == true) and 1 or 0
	j = (y == true) and 1 or 0
	input = torch.Tensor({i,j})
	k.build({2,2,1})
	lay = k.getLayer(1)
	lay[{{},{}}]= torch.Tensor{{-10,20,20},{30,-20,-20}}
	la = k.getLayer(2)
	la[1] = torch.Tensor{-30,20,20}
	r = k.forward(input)
	return (r[1]>0.5) and true or false
end

return M

