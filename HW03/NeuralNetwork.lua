local network = {}
local Theta = {}
local dE_dTheta ={}
local a ={}
local z ={}
local delta = {}
require 'math'


function network.build(a)
	for i = 1, #a-1 do
		network[i] = torch.randn(a[i+1],a[i]+1)/math.sqrt(a[i])
		Theta[i] = network[i][{{},{2,-1}}]
	end
end

local function sigmoid(b)
	for i = 1, b:size(1) do 
		b[i] = 1/(1+math.exp(-b[i]))
	end
	return b
end

function network.getLayer(k)
	return torch.cat(network[k][{{},{1}}],Theta[k])
end

function network.forward(input)
	res = input
	bias = torch.Tensor(1):fill(1)
	a[1] = res
	z[1] = res
	for i = 1, #network do	

		res = network[i]*torch.cat(bias,res)
		z[i+1] = res
		res = sigmoid(res)
		a[i+1] = res
	end 
	return res
end

function network.backward(target)
	delta[#network+1] = torch.cmul(a[#network+1] - target,torch.cmul(a[#network+1],1-a[#network+1]))
	for i = #network, 1, -1 do

		delta[i] = torch.cmul(Theta[i]:t() * delta[i+1],torch.cmul(a[i],1-a[i]))
	end
	for i = 1, #network do
		dE_dTheta[i] = a[i]:view(-1,1)*delta[i+1]:view(1,-1)
		dE_dTheta[i] = dE_dTheta[i]:t()
	end
end

function network.updateParams(etha)
	for i = 1, #network do
		Theta[i] = Theta[i] - etha*dE_dTheta[i]
		network[i][{{},{2,-1}}] = Theta[i]
	end
	
end
return network


