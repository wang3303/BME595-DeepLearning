local mnist = require 'mnist'
local trainset = mnist.traindataset()
local testset = mnist.testdataset()
k = require 'NeuralNetwork'
local M = {}

function M.train()
	k.build({784,100,10})
	co = torch.Tensor(60000):fill(0)
	i =1
	co[i] = 1000
	while(co[i] > 0.1) do	
		i = i+1
		input = trainset[i].x:double()
		input = input:resize(784)
		target = torch.Tensor(10):fill(0)
		if trainset[i].y == 0 then
			num = 10
			else num = trainset[i].y
		end
		target[num] = 1
		
		
		k.forward(input)

		k.backward(target)
		k.updateParams(0.08)
		
		for j = 1,500 do
			input = testset[i].x:view(784)
			input = input:double()
			if trainset[i].y == 0 then
				num = 10
				else num = trainset[i].y
			end
			target[num] = 1
			res = k.forward(input)
			res = res:resize(10)
			co[i] = co[i]+0.5*torch.sum((target-res):pow(2)) --J()
			
		end
		p.plot(co[{{1,i}}])
		--print(co:min())
	end

	
end

function forward(img)
	
end
return M
