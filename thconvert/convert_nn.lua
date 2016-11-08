require 'nn'
require 'cutorch'

cmd = torch.CmdLine()
cmd:option('-model','','input model')
cmd:option('-classes','','classes t7')
cmd:option('-modelOut','','output model')

params = cmd:parse(arg)

if params.model=='' then
	print("No model specified!")
	os.exit(1)
end

if params.classes=='' then
	print("no clases t7 specificed")
	os.exit(1)
end

function convert(main_module) 
	if torch.typename(main_module)=='nn.Sequential' then
		local new_main_module=nn.Sequential()
		--we should iterate over it?
		for i=1,#main_module.modules do
			local module = main_module.modules[i]
			local ty = torch.typename(module)
			if torch.typename(module)=='nn.Sequential' then
				new_main_module:add(convert(module))
			elseif ty=='nn.SpatialBatchNormalization' or ty=='nn.BatchNormalization' then
				local c = main_module.modules[i-1]
				for i=1,c.weight:size()[1] do
					local mdivv = module.weight[i]/torch.sqrt(module.running_var[i]+module.eps)
					--scale both
					c.weight[i]:mul(mdivv)
					c.bias[i]=(c.bias[i]-module.running_mean[i])*mdivv+module.bias[i]
				end
			elseif ty=='nn.ReLU' then
				new_main_module:add(nn.ReLU(false))			
			elseif ty=='nn.Dropout' then
				--v2 dropout doesnt need us to do anythign!
			else
				new_main_module:add(module)			
			end
		end
		return new_main_module
	end
	return main_module
end

function flatten_model(model,new_model)
	--check if all are sequential
	if new_model==nil then
		new_model=nn.Sequential()
	end
	for i=1,#model.modules do
		local module = model.modules[i]
		local ty = torch.typename(module)
		if torch.typename(module)=='nn.Sequential' then
			flatten_model(module,new_model)
		else
			new_model:add(module)
		end
	end
	return new_model
end

torch.manualSeed(0)
--local d = torch.rand(3,224,224):fill(0.1)
require 'image'
--local d = image.crop(image.scale(image.lena(),256,256),"c",224,224)
local d = image.crop(image.load('test.jpg',3,'double'),"c",224,224)*255
--print(d)
--print("IMAGE SUM",d:sum())
local model = torch.load(params.model):double()
local classes = torch.load(params.classes)

model:evaluate()

local new_model = flatten_model(convert(model:clone()))
new_model.classes = classes
--print("CLASSES",new_model.clases)

print(model:forward(d))
print(new_model:forward(d))

--print("IN VALUE",d:sum())
local debug=false
if debug then
	for i=1,#new_model.modules do
		local m = new_model.modules[i]
		if torch.typename(m)=='nn.SpatialConvolution' then
			--print(m.output:size(),m.bias:sum(),m.output:sum())
			print(m.output:sum())
		else
			print(torch.typename(m),m.output:sum())
			if (m.output:nElement()<10) then
				print(m.output)
			end
		end
	end
end

new_model:float()
if params.modelOut~='' then
	torch.save(params.modelOut,new_model)
end

test=false
if test then
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

	local mm = convert(m)

	print(m)
	print(mm)

	local out_cbn = mm:forward(d)

	print((out_bn-out_cbn):sum())
end
