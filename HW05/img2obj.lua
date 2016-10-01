require 'nn'
require 'math'
require 'torch'
require 'image'
require 'camera'
local p = require 'gnuplot'
local trainset = torch.load('cifar100-train.t7')
local obj = {}
local etha = 0.006
trainset.data = trainset.data:double()
--setmetatable
setmetatable(trainset,{__index = function(t,i) return {t.data[i],t.label[i]} end})


local mean = {}
local std = {}


--build network
local net=nn.Sequential()

--first layer
net:add(nn.SpatialConvolution(3,6,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.ReLU())
--second layer
net:add(nn.SpatialConvolution(6,16,5,5))
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.ReLU())

net:add(nn.View(16*5*5))

--Linear
net:add(nn.Linear(16*5*5,200))
net:add(nn.ReLU())
net:add(nn.Linear(200,100))
net:add(nn.ReLU())
net:add(nn.Linear(100,100))

--cost function
local loss = nn.MSECriterion()

--functions
local function oneHot(num)
	target = torch.Tensor(100):fill(0)
		if num == 0 then
			num = 100
		end
		target[num] = 1
	return target
end

function obj.train()
	--normalization
	for i = 1, 3 do 
		mean[i] = trainset.data[{{},{i},{},{}}]:mean()
		trainset.data[{{},{i},{},{}}]:add(-mean[i])
		std[i] = trainset.data[{{},{i},{},{}}]:std()
		trainset.data[{{},{i},{},{}}]:div(std[i])
	end
	d = torch.Tensor(1000):fill(0)
	batchsize = 50
	shuffle = torch.Tensor(batchsize)
	maxIteration = 50
	for i = 1, maxIteration do
		shuffle:random(50000)
		net:zeroGradParameters()
		for j = 1, batchsize do
			set = trainset[i]
			input = set[1]
			output = oneHot(set[2])
			local pred = net:forward(input)

			err = loss:forward(pred,output) 
			local gradloss = loss:backward(pred,output)
			net:backward(input,gradloss)
		end
		net:updateParameters(etha)
		d[i] = err
		if err < 0.001 then break end
		--p.plot(d[{{1,i}}])	
	end
	
	
end

function obj.forward(img)
	output = net:forward(img:double())
	max = output[1]
	res = 1
	for i = 2, 100 do
		if max<output[i] then
			res = i
			max = output[i]
		end
	end
	res = (res == 100) and 0 or res
	return tostring(res)
end

function obj.view(img)
	k = image.drawText(img,obj.forward(img),10,10,{color = {0, 255, 0}})
	image.display{image = k, legend = "identify the picture"}
end

function obj.cam()
	local cam = image.Camera{}
	local frame = cam:forward()
	local w = image.display{image = frame, legend = "identify the picture"}
	for i = 1,1000 do
	print(i)
		frame = cam:forward()
		frame = image.scale(frame,32,32)
		frame:double()
		str = obj.forward(frame)
		k = image.drawText(frame,str,10,10,{color = {0, 255, 0}})
		image.display{image = k, legend = "identify the picture",win =w}
	end
	cam:stop()
end

return obj

