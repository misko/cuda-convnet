require 'nn'

local fs=3
local from=5
local to=7
local m = nn.Sequential()
local c = nn.SpatialConvolution(from,to,fs,fs,1,1)
local bn = nn.SpatialBatchNormalization(to,0)
local d = torch.rand(2,from,fs,fs)

m:add(c):add(bn)

m:evaluate()

local out_bn = m:forward(d)

mm=nn.Sequential()
for i,module in ipairs(m:listModules()) do
	if torch.typename(module)=='nn.Sequential' then
			
	elseif torch.typename(module)=='nn.SpatialBatchNormalization' then
		local c = m:listModules()[i-1]
		for i=1,c.weight:size()[1] do
			local mdivv = module.weight[i]/torch.sqrt(module.running_var[i])
			--scale both
			c.weight[i]:mul(mdivv)
			c.bias[i]=c.bias[i]*mdivv+module.bias[i]
		end
	else	
		mm:add(module)
	end
end

local out_cbn = mm:forward(d)

print((out_bn-out_cbn):sum())
