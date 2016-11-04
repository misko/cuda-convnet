import torchfile
import sys

def get_layer(model,layer_type,ly):
	for i in range(len(model.modules)):
		l =  model.modules[i]
		ty = l.torch_typename()
		if ty==layer_type:
			if ly[0]==1:
				return (l.weight, l.bias)
			ly[0]-=1
		elif ty=='nn.Sequential':
			r=get_layer(l,layer_type,ly)
			if r!=None:
				return r
	return None

def get_layer_from_file(fn,layer_type,ly):
	model = torchfile.load(model_fn)
	return get_layer(model,layer_type,[ly])

if __name__=='__main__':
	if len(sys.argv)!=4:
		print "%s model_filename layer_type layer" % sys.argv[0]
		exit(1)

	model_fn = sys.argv[1]
	layer_type = sys.argv[2]
	layer_n = int(sys.argv[3])

	l=get_layer_from_file(model_fn,layer_type,layer_n)
	if l:
		print l[0].shape,l[1].shape
