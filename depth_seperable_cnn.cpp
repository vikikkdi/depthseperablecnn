//This is the cpp implementation for depthwise seperable convolutional neural network - forward pass
#include<bits/stdc++.h>
using namespace std;


float rand_float(){
	return (float)rand()/(float)(RAND_MAX);
}

vector<vector<vector<float> > > zero_pad(vector<vector<vector<float> > > inp, int c, int h, int w, int k_h, int k_w){
	int p1 = (int)(k_h-1)/2;
	int p2 = (int)(k_w-1)/2;
	int out_h = 2*p1+h;
	int out_w = 2*p2+w;

	vector<vector<vector<float> > > out (c,vector<vector<float> >(out_h,vector <float>(out_w)));
	
	for(int i=0;i<c;i++){
		for(int j=0;j<out_h;j++){
			for(int k=0;k<out_w;k++){
				out[i][j][k] = 0;
			}
		}
	}

	for(int i=0;i<c;i++){
		for(int j=0;j<h;j++){
			for(int k=0;k<w;k++){
				out[i][p1+j][p2+k] = inp[i][j][k];
			}
		}
	}

	return out;
}

float multiply(vector<vector<float> > a, vector<vector<float> > b, int n, int m){
	float val = 0;
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			val += a[i][j]*b[i][j];
		}
	}
	return val;
}

vector<vector<float> > slice_vec(vector<vector<float> > a, int sx, int sy, int h, int w){
	vector<vector<float> > out(h, vector<float>(w));
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			out[i][j] = a[sx+i][sy+j];
		}
	}
	return out;
}

vector<vector<vector<float> > > seperable_conv2d(int c, int h, int w, int k_h, int k_w, int f, vector<vector<vector<float> > > inp, vector<vector<vector<float> > > kweight, vector<vector<float > > pweight, int stride){
	vector<vector<vector<float> > > padded = zero_pad(inp, c, h, w, k_h, k_w);
	int p1 = (int)(k_h-1)/2;
	int p2 = (int)(k_w-1)/2;
	int out_h = (int)((h-k_h+2*p1)/(stride)+1);
	int out_w = (int)((w-k_w+2*p2)/(stride)+1);

	//depthwise separable convolution separates convolution process into 2 parts: a depthwise convolution and a pointwise convolution.

	/*
		step1: Depth wise convolution
		input shape is c*h*w eg 3*12*12
		kernal shape is k_h*k_w eg 5*5
		output shape is c*out_h*out_w eg 3*8*8
	*/

	vector<vector<vector<float> > > out(c,vector<vector<float> >(out_h,vector <float>(out_w)));
	for(int i=0;i<c;i++){
		int sx = 0;
		for(int j=0;j<out_h;j++){
			int sy = 0;
			for(int k=0;k<out_w;k++){
				out[i][j][k] = multiply(slice_vec(padded[i], sx, sy, k_h, k_w), kweight[i], k_h, k_w);
				sy += stride;
			}
			sx += stride;
		}
	}
	/*
		step2: point wise convolution
		input shape is c*out_h*out_w eg 3*8*8
		point kernel shape is f*c eg 256*3
		output shape is f*out_h*out_w eg 256*8*8
	*/

	vector<vector<vector<float> > > final_out(f,vector<vector<float> >(out_h,vector <float>(out_w)));
	for(int i=0;i<f;i++){
		for(int j=0;j<out_h;j++){
			for(int k=0;k<out_w;k++){
				final_out[i][j][k] = 0;
			}
		}
	}

	for(int i=0;i<f;i++){
		for(int j=0;j<out_h;j++){
			for(int k=0;k<out_w;k++){
				for(int l=0;l<c;l++){
					final_out[i][j][k] += out[l][j][k]*pweight[i][l];
				}
			}
		}
	}
	return final_out;

}

int main(){
	int c = 3, h = 12, w = 12, k_h = 5, k_w = 5, f = 256, stride = 1;
	vector<vector<vector<float> > > inp (c,vector<vector<float> >(h,vector <float>(w)));
	vector<vector<vector<float> > > kweight (c,vector<vector<float> >(k_h,vector <float>(k_w)));
	vector<vector<float> > pweight(f, vector<float>(c));
	
	//initialise input matrix with random weights
	for(int i=0;i<c;i++){
		for(int j=0;j<h;j++){
			for(int k=0;k<w;k++){
				inp[i][j][k] = (int)(rand_float()*100);
			}
		}
	}
	

	//initialise kweight matrix with random weights
	for(int i=0;i<c;i++){
		for(int j=0;j<k_h;j++){
			for(int k=0;k<k_w;k++){
				kweight[i][j][k] = (rand_float()*10);
			}
		}
	}

	//initialise pweight matrix with random weights
	for(int i=0;i<f;i++){
		for(int j=0;j<c;j++){
			pweight[i][j] = (rand_float()*10);
		}
	}

	vector<vector<vector<float> > > out = seperable_conv2d(c, h, w, k_h, k_w, f, inp, kweight, pweight, stride);

	for(int i=0;i<f;i++){
		for(int j=0;j<h;j++){
			for(int k=0;k<w;k++){
				cout<<out[i][j][k]<<" ";
			}
			cout<<endl;
		}
		cout<<endl<<endl;
	}

}