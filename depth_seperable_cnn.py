'''
	This is the python implementation for depthwise seperable convolutional neural network - forward pass
'''

import random as rand

def zero_pad(inp, c, h, w, k_h, k_w):
	'''
		padding the input array using SAME padding where we add equal number of 
		zeros for both row and column
	'''
	p1, p2 = int((k_h-1)/2), int((k_w-1)/2)
	out_h, out_w = 2*p1+h, 2*p2+w
	out = []

	for i in range(c):
		x = []
		for i in range(out_h):
			y = [0 for _ in range(out_w)]
			x.append(y)
		out.append(x)
	
	for i in range(c):
		for j in range(h):
			for k in range(w):
				out[i][p1+j][p2+k] = inp[i][j][k]
	
	return [out, p1, p2]

def multiply(inp, kweight, k_h, k_w):
	val = 0
	for i in range(k_h):
		for j in range(k_w):
			val = val + (inp[i][j]*kweight[i][j])
	return val

def slice(inp, sx, sy, h, w):
	out = []
	for i in range(h):
		x = []
		for j in range(w):
			x.append(inp[sx+i][sy+j])
		out.append(x)
	return out

def seperable_conv2d(c, h, w, k_h, k_w, f, inp, kweight, pweight, stride):
	'''
		c is the number of channels in the given input
		h and w are the height and width of the input matrix
		k_h and k_w are the height and width of the kernel matrix
		f is the number of kernels to be used
		input is the matrix that is given as the input
		kweight is the matrix which consists of the weights for the kernel of size k_h * k_w
		pweight is the matrix which consists of the weights f for 1*1*c kernel
	'''

	#depthwise separable convolution separates convolution process into 2 parts: a depthwise convolution and a pointwise convolution.

	'''
		step1: Depth wise convolution
		input shape is c*h*w eg 3*12*12
		kernal shape is k_h*k_w eg 5*5
		output shape is c*out_h*out_w eg 3*8*8
	'''
	inp, p1, p2 = zero_pad(inp, c, h, w, k_h, k_w)
	out_h, out_w = ((h-k_h+2*p1)/(stride)+1), ((w-k_w+2*p2)/(stride)+1)
	out_h, out_w = int(out_h), int(out_w)
	
	out = []

	for i in range(c):
		x = []
		for i in range(out_h):
			y = [0 for _ in range(out_w)]
			x.append(y)
		out.append(x)

	for _ in range(c):
		sx = 0
		for j in range(out_h):
			sy = 0
			for k in range(out_w):
				out[_][j][k] = multiply(slice(inp[_], sx, sy, k_h, k_w), kweight[_], k_h, k_w)
				sy += stride
			sx += stride

	'''
		step2: point wise convolution
		input shape is c*out_h*out_w eg 3*8*8
		point kernel shape is f*c eg 256*3
		output shape is f*out_h*out_w eg 256*8*8
	'''
	final_out = []
	for i in range(f):
		x = []
		for i in range(out_h):
			y = [0 for _ in range(out_w)]
			x.append(y)
		final_out.append(x)

	for _ in range(f):
		for i in range(out_h):
			for j in range(out_w):
				for k in range(c):
					final_out[_][i][j] += out[k][i][j]*pweight[_][k]

	return final_out


if __name__=='__main__':

	c = 3
	h = 12
	w = 12
	k_h = 5
	k_w = 5
	f = 256
	inp = []
	kweight = []
	pweight = []
	stride = 1

	'''
		Initialize the entries for the input of size h*w*c using random values
		It is of the format inp[c][h][w]
	'''
	for i in range(c):
		x = []
		for j in range(h):
			y = []
			for k in range(w):
				y.append(int(rand.random()*100))
			x.append(y)
		inp.append(x)

	'''
		Initialize the entries for the kweight of size k_h*k_w*c using random values
		It is of the format kweight[c][k_h][k_w]
	'''
	for i in range(c):
		x = []
		for j in range(k_h):
			y = []
			for k in range(k_w):
				y.append(rand.random()*10)
			x.append(y)
		kweight.append(x)

	'''
		Initialize the entries for the pweight of size c*f using random values
		It is of the format pweight[f][c]
	'''
	for i in range(f):
		x = []
		for j in range(c):
			x.append(rand.random()*10)
		pweight.append(x)

	out = seperable_conv2d(c, h, w, k_h, k_w, f, inp, kweight, pweight, stride)
	print(out)