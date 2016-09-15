local network = {}
require 'math'


function network.build(a)
	for i = 1, #a-1 do
		network[i] = torch.randn(a[i+1],a[i]+1)/math.sqrt(a[i])
	end

end

local function sigmoid(b)
	for i = 1, b:size(1) do 
		b[i] = 1/(1+math.exp(-b[i]))
	end
	return b
end

function network.getLayer(k)
	return network[k]
end

function network.forward(input)
	res = input
	bias = torch.Tensor(1):fill(1)
	for i = 1, #network do
		
		res = network[i]*torch.cat(bias,res,1)
		 res = sigmoid(res)
	end
	return res
end

function backward()

end

function updateParams()

end
return network

