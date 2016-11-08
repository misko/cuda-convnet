import torchfile
import sys
import numpy as np
import binary
import math

def get_layer_from_model(model,layer_type,ly):
	for i in range(len(model.modules)):
		l =  model.modules[i]
		ty = l.torch_typename()
		if ty==layer_type:
			if ly[0]==1:
				return {'weight':l.weight,'bias':l.bias,'kW':l.kW,'kH':l.kH,'padW':l.padW,'padH':l.padH,'dW':l.dW,'dH':l.dH,'channels':l.nInputPlane,'filters':l.nOutputPlane}
			ly[0]-=1
		elif ty=='nn.Sequential':
			r=get_layer_from_model(l,layer_type,ly)
			if r!=None:
				return r
	return None

def flip(m):
	return np.flipud(np.fliplr(m))

def get_layer_from_file(fn,layer_type,ly):
	model = torchfile.load(fn)
	return get_layer_from_model(model,layer_type,[ly])

def get_layer_test(name,idx,shapes,params):
	print name,idx,shapes
	l=get_layer_from_file("model.t7","nn.SpatialConvolution",1)
	print l['weight'].shape,l['bias'].shape
	exit(1)

def get_type(s):
	if s[:4]=='conv':
		return ('nn.SpatialConvolution',int(s[4:]))
	elif s[:2]=='fc':
		return ('nn.Linear',int(s[2:]))
	return None	

def get_layer(name,fn):
	ty=get_type(name)
	if ty==None:
		print "Unknown type!",name
		exit(1)
	d=get_layer_from_file(fn,ty[0],ty[1])
	return d

def weights_torch_to_convnet(w):
	#convnet
	#weights.resize((input_channels, ksize, ksize, total_num_kernels))
	#torch
	#(OUT/IN/FS/FS) - self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
	w=w.copy()
	flipc=False
	if flipc:
		for f in range(w.shape[0]):
			for c in range(w.shape[1]):
				w[f,c,:,:]=flip(w[f,c,:,:])
	#return w.copy().transpose(1,3,2,0)
	return w.copy().transpose(1,2,3,0)

def get_layer_weight(name,idx,shapes,params):
	if len(params)==0:
		print "get_layer_weight failed, no params!",name
		exit(1)
	l=get_layer(name,params[0])
	print(l['weight'].shape)
	if l['weight'].ndim==2:
		w=l['weight'].transpose(1,0)
		print "LAYER INIT FORM FILE",w.shape,shapes
		return w.reshape(shapes)
	else:
		w=l['weight'].copy()
		#for i in range(w.shape[0]):
		#	for j in range(w.shape[1]):
		#		w[i,j,:,:]=flip(w[i,j,:,:])
				
		#w=w.transpose(1,2,3,0)
		w=weights_torch_to_convnet(w)
		print "LAYER INIT FORM FILE",w.shape,shapes
		return w.reshape(shapes)

def get_layer_bias(name,shapes,params):
	if len(params)==0:
		print "get_layer_weight failed, no params!",name
		exit(1)
	l=get_layer(name,params[0])
	b=l['bias']
	return b.reshape(shapes)

def write_fc(name,d):
	w = d['weight'].transpose(1,0)
	b = d['bias']
	print 'FC weights.shape=%s' % (str(w.shape))
	#torch format is (out,in) for linear, and something like ( out, chan*fs*fs) when it goes through a view
	#convnet format is (in,out)
	outw=w
	if 'channels' in d:
		sz=int(math.sqrt(d['weight'].shape[1]/d['channels']))
		#transpose makes, (out,chan*fs*fs) to (chan*fs*fs,out)
		w = w.reshape((d['channels'], sz, sz, d['weight'].shape[0]))
		outw = np.empty((sz, sz, d['channels'], d['weight'].shape[0]), dtype=d['weight'].dtype)
		for channel in range(d['channels']):
			outw[:, :, channel, :] = w[channel, :, :, :]
		print "FC <> ",outw.shape
		outw.resize(d['weight'].shape[1], d['weight'].shape[0])

	#print 'converted_weights.shape=%s' % (str(outw.shape))
	
	payload = bytearray()
	payload.extend(binary.to_string('class'))
	payload.extend(binary.to_string('neuron'))
	payload.extend(binary.to_string('name'))
	payload.extend(binary.to_string(name))
	spec = {'num_output': outw.shape[1]}
	payload.extend(binary.to_string('spec'))
	payload.extend(binary.convert_simple_dict(spec))
	payload.extend(binary.to_string('weight'))
	payload.extend(binary.numpy_array_to_binary(outw))
	payload.extend(binary.to_string('has_bias'))
	payload.extend(binary.to_uint32(1))
	payload.extend(binary.to_string('bias'))
	payload.extend(binary.numpy_array_to_binary(b))
	payload.extend(binary.to_string('dropout'))
	payload.extend(binary.to_float32(0.0))
	output = binary.to_dict(payload)
	return output

def write_relu(name):
	payload = bytearray()
	payload.extend(binary.to_string('class'))
	payload.extend(binary.to_string('relu'))
	payload.extend(binary.to_string('name'))
	payload.extend(binary.to_string(name))
	output = binary.to_dict(payload)
	return output

def write_softmax(name):
	payload = bytearray()
	payload.extend(binary.to_string('class'))
	payload.extend(binary.to_string('max'))
	payload.extend(binary.to_string('name'))
	payload.extend(binary.to_string(name))
	output = binary.to_dict(payload)
	return output

def write_pool(name,d):
	payload = bytearray()
	assert(d['padW']==0)
	assert(d['mode']=='max')
	payload.extend(binary.to_string('class'))
	payload.extend(binary.to_string('pool'))
	payload.extend(binary.to_string('name'))
	payload.extend(binary.to_string(name))
	payload.extend(binary.to_string('psize'))
	payload.extend(binary.to_uint32(d['kW']))
	payload.extend(binary.to_string('stride'))
	payload.extend(binary.to_uint32(d['dW']))
	payload.extend(binary.to_string('mode'))
	payload.extend(binary.to_string('max'))
	output = binary.to_dict(payload)
	return output


def layer_to_pool(l):
	return {'ty':'pool','kW':l.kW,'kH':l.kH,'padW':l.padW,'padH':l.padH,'dW':l.dW,'dH':l.dH}

def layer_to_fc(l):
	return {'ty':'fc', 'weight':l.weight, 'bias':l.bias, 'inputs':l.weight.shape[1], 'outputs':l.weight.shape[0]}

def layer_to_conv(l):
	return {'ty':'conv','weight':l.weight,'bias':l.bias,'kW':l.kW,'kH':l.kH,'padW':l.padW,'padH':l.padH,'dW':l.dW,'dH':l.dH,'channels':l.nInputPlane,'filters':l.nOutputPlane}

def write_model(model,filename):
	layers = bytearray()
	parsed_layers = []
	for i in range(len(model.modules)):
		i_s=str(i)
		l =  model.modules[i]
		ty = l.torch_typename()
		d=None
		if ty=="nn.SpatialConvolution":
			d=layer_to_conv(l)
			layers.extend(write_conv("conv"+i_s,d))
			parsed_layers.append(d)
		elif ty=="nn.ReLU":
			layers.extend(write_relu("relu"+i_s))
		elif ty=="nn.SpatialMaxPooling":
			d=layer_to_pool(l)
			d['filters']=parsed_layers[-1]['filters']
			d['channels']=parsed_layers[-1]['filters']
			d['mode']='max'
			assert(d['padW']==0)
			layers.extend(write_pool("pool"+i_s,d))
			parsed_layers.append(d)
		elif ty=="nn.View":
			pass
		elif ty=="nn.Linear":
			#torch format is (out,in) for linear, and something like ( out, chan*fs*fs) when it goes through a view
			#convnet format is (in,out)
			d=layer_to_fc(l)
			if parsed_layers[-1]['ty']!='fc':
				d['channels']=parsed_layers[-1]['filters']
				#ok lets resize the weights!
				layers.extend(write_fc("fc"+i_s,d))
			else:
				layers.extend(write_fc("fc"+i_s,d))
			parsed_layers.append(d)
		elif ty=="nn.LogSoftMax":
			if parsed_layers[-1]['ty']!='fc':
				print "LOGSOFTMAX MUST FOLLOW FC?"
				exit(1)
			d={'ty':'logsoftmax','outputs':parsed_layers[-1]['outputs']}
			layers.extend(write_softmax("probs"))
			parsed_layers.append(d)
		else:
			print "Layer Unsupported!!"
			exit(1)
	
	graph = bytearray()
	graph.extend(binary.to_string('layers'))
	graph.extend(binary.to_list(layers))

	print "PUTTING IN FAKE ZERO MEAN!"
	#data_mean = self.train_data_provider.batch_meta['data_mean'].astype(n.float32)
	data_mean = np.empty((256,256,3)).astype(np.float32)
        data_mean.fill(0)
	graph.extend(binary.to_string('data_mean'))
	graph.extend(binary.numpy_array_to_binary(data_mean))
	
	labels_payload = bytearray()
	#label_names = self.train_data_provider.batch_meta['label_names']
	#label_names=map( lambda x : str(x), range(parsed_layers[-1]['outputs']))
	#for label_name in label_names:
	#	labels_payload.extend(binary.to_string(label_name))
	for label_name in model['classes']:
		labels_payload.extend(binary.to_string(label_name))
	graph.extend(binary.to_string('label_names'))
	graph.extend(binary.to_list(labels_payload))

	output = binary.to_dict(graph)
	file = open(filename, 'wb')
	file.write(output)
	file.close()

#takes torch format and writes to conv payload
def write_conv(name,d):
	w=weights_torch_to_convnet(d['weight'])
	b=d['bias']
	print "CONV input_channels=%s, ksize=%s, total_num_kernels=%s %e" % (str(d['channels']), str(d['kW']), str(d['filters']),w.sum())

	#w.resize((channels, ksize, ksize, kernels))
	outw = np.empty((d['kW'], d['kW'], d['channels'], d['filters']), dtype=w.dtype)
	for i in range(d['channels']):
	  outw[:, :, i, :] = w[i, :, :, :]
	outw.resize(d['kW'] * d['kH'] * d['channels'], d['filters'])
	#sys.stderr.write('outw.shape=%s\n' % (str(outw.shape)))
	#sys.stderr.write('b.shape=%s\n' % (str(b.shape)))
	
	payload = bytearray()
	payload.extend(binary.to_string('class'))
	payload.extend(binary.to_string('conv'))
	payload.extend(binary.to_string('name'))
	payload.extend(binary.to_string(name))
	payload.extend(binary.to_string('spec'))
	spec = {
	  'num_kernels': d['filters'],
	  'ksize': d['kW'],
	  'stride': d['dW'],
	}
	payload.extend(binary.convert_simple_dict(spec))
	payload.extend(binary.to_string('kernels'))
	payload.extend(binary.numpy_array_to_binary(outw))
	payload.extend(binary.to_string('has_bias'))
	payload.extend(binary.to_uint32(1))
	payload.extend(binary.to_string('bias'))
	payload.extend(binary.numpy_array_to_binary(b))
	payload.extend(binary.to_string('padding'))
	payload.extend(binary.to_uint32(d['padW']))
	output = binary.to_dict(payload)
	return output

	

if __name__=='__main__':
	if len(sys.argv)!=3:
		print "%s model_filename outfile" % sys.argv[0]
		exit(1)

	model_fn = sys.argv[1]
	out_fn = sys.argv[2]

	model = torchfile.load(model_fn)
	write_model(model,out_fn)


	#d=get_layer_from_file(model_fn,layer_type,layer_n)
	#if d:
	#	print d['weight'].shape,d['bias'].shape
	#	write_conv("test",d)
		
