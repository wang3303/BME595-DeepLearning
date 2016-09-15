local M = {}

function M.AND(x,y)
	if x==true and y==true then
		return true 
		else return false
		end
end

function M.OR(x,y)
	if x==false and y==false then
		return false
		else return true
end
end

function M.NOT(x)
	if x == false 
		then return true 
		else return false
		end
	end

function M.XOR(x,y)
	if M.NOT(x) == y 
		then return true 
		else return false
		end
end

return M

