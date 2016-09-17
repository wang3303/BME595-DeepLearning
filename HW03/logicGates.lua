k = require 'NeuralNetwork'
p = require 'gnuplot'
require 'torch'
require 'math'

local M = {}
local set = torch.Tensor({{0,1},{1,0},{0,0},{1,1}})
local xor = torch.Tensor{1,1,0,0}

AND = {}
OR = {}
NOT = {}
XOR = {}

local function tr(k)
	return (k == true) and 1 or 0
end

local function dw(k)
	return (k == 1) and true or false
end

function AND.train()
co = torch.Tensor(150000):fill(0)
	k.build({2,1,1})
	i = 1
	co[i] = 1
	while (co[i] > 0.01 and i < 150000) do
		i = i+1		
		co[i] = 0
		c = set[torch.random(4)]
		target = tr(dw(c[1]) and dw(c[2]))
		
		
		r = k.forward(c)

		k.backward(target)
		k.updateParams(0.03)
		
		for j = 1,4 do
			c = set[j]
			target = c[1]*c[2]
			res = k.forward(c)
			co[i] = co[i]+0.5*math.pow(target-res[1],2) --J()
		end
		
	end
 	print(k.getLayer(1))
	print(k.getLayer(2))
	for j = 1,4 do
			c = set[j]
			print(c)
			print(k.forward(c))
		end
	p.plot(co)
end

function OR.train(x,y)
	co = torch.Tensor(150000):fill(0)
	k.build({2,1,1})
	i = 1
	co[i] = 1
	while (co[i] > 0.1 and i < 150000) do
		i = i+1		
		co[i] = 0
		c = set[torch.random(4)]
		target = tr(dw(c[1]) or dw(c[2]))
		
		
		r = k.forward(c)

		k.backward(target)
		k.updateParams(0.3)
		
		for j = 1,4 do
			c = set[j]
			target = tr(dw(c[1]) or dw(c[2]))
			res = k.forward(c)
			co[i] = co[i]+0.5*math.pow(target-res[1],2)
		end
		
	end
 	print(k.getLayer(1))
	print(k.getLayer(2))
	for j = 1,4 do
			c = set[j]
			print(c)
			print(k.forward(c))
		end
	p.plot(co)
end

function NOT.train()
	co = torch.Tensor(150000):fill(0)
	k.build({1,1})
	i = 1
	co[i] = 1
	while (co[i] > 0.01 and i < 150000) do
		i = i+1		
		co[i] = 0
		c = set[torch.random(2)]
		target = torch.Tensor(1):fill(c[2])
		
		
		r = k.forward(torch.Tensor(1):fill(c[1]))

		k.backward(target)
		k.updateParams(0.05)
		
		for j = 1,2 do
			c = set[j]
			target = c[2]
			res = k.forward(torch.Tensor(1):fill(c[1]))
			co[i] = co[i]+0.5*math.pow(target-res[1],2)
		end
		
	end
 	print(k.getLayer(1))
	for j = 1,2 do
			c = set[j]
			print(c)
			print(k.forward(torch.Tensor(1):fill(c[1])))
		end
	p.plot(co)
end

function XOR.train()
	co = torch.Tensor(150000):fill(0)--set a limit to see the trend
	k.build({2,2,1})
	i = 1
	co[i] = 1
	while (co[i] > 0.05 and i < 150000) do
		i = i+1		
		co[i] = 0
		rand = torch.random(4)
		c = set[rand]
		target = xor[rand]
		
		
		r = k.forward(c)

		k.backward(target)
		k.updateParams(0.08)
		
		for j = 1,4 do
			c = set[j]
			target = xor[j]
			res = k.forward(c)
			co[i] = co[i]+0.5*math.pow(target-res[1],2)
		end
		
	end
 	print(k.getLayer(1))
	print(k.getLayer(2))
		for j = 1,4 do
			c = set[j]
			print(c)
			print(k.forward(c))
		end
	p.plot(co)
end
return M


