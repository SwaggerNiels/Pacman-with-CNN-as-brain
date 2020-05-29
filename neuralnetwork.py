# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:19:35 2020

@author: niels
"""
import numpy as np;
from random import shuffle;
from scipy import signal;
from skimage.measure import block_reduce;

class NeuralNetwork:
    def __init__(self, layer_sizes):
        '''Set all the neural network properties.'''
        
        #global shape
        self.layer_sizes = layer_sizes;
        self.L = len(layer_sizes);
        self.weight_shapes = [(a,b) for a,b in zip(layer_sizes[1:],layer_sizes[:-1])];
        
        #weights and biases
        self.w = [np.random.standard_normal(s)/s[1]**.5 for s in self.weight_shapes];
        self.b = [np.random.standard_normal((s,1)) for s in layer_sizes[:]];
        
        #z-values, activation values and error values
        self.z = [np.zeros((s,1)) for s in layer_sizes[:]];
        self.a = [np.zeros((s,1)) for s in layer_sizes[:]];
        self.d = [np.zeros((s,1)) for s in layer_sizes[:]];
        
    def printShapes(self):
        '''Print out the attribute shapes per layer'''
        for l in range(self.L-1):
            print(f"layer {l}");
            print(f"b: {self.b[l].shape}");
            print(f"w: {self.w[l].shape}");
            print(f"a: {self.a[l].shape}");
            print(f"z: {self.z[l].shape}");
            print(f"d: {self.d[l].shape}");
        l = self.L-1;
        print(f"layer {l}");
        print(f"b: {self.b[l].shape}");
        print(f"a: {self.a[l].shape}");
        print(f"d: {self.d[l].shape}");
        
    def feedForward(self, x):
        '''Returns the output of output layer after given input to input layer.'''
        
        self.a[0] = x+self.b[0];
        
        for l in range(1,self.L):
            z_l = np.matmul(self.w[l-1],self.a[l-1]) + self.b[l];
            self.z[l] = z_l;
            self.a[l] = self.sigma(z_l);

        return(self.a[self.L-1]);

    def backPropagate(self, y):
        '''Provided a feedForward is executed with input 'x' this function will
        backpropagate the errors in output relative to the labeling 'y'. This
        function will return the 2 lists: one for the updates biases and one
        for the weights.'''
        
        self.d[-1] = (self.a[-1] - y) * self.sigmaPrime(self.z[-1]);
        for l in reversed(range(1,self.L-1)):
            self.d[l] = np.matmul(self.w[l].T,self.d[l+1]) * self.sigmaPrime(self.z[l]);
        
        #record the change in the weights and biases based on the error, d
        update_b = [np.zeros(b.shape) for b in self.b];
        update_w = [np.zeros(w.shape) for w in self.w];
        
        update_b[0] = self.d[0];
        for l in range(1,self.L):            
            update_b[l] = self.d[l];
            update_w[l-1] = self.a[l-1].T * self.d[l];
        
        return(update_b,update_w);

    def trainNetwork(self, training_pairs, rate):
        '''Provided some training pairs consisting of input 'x' and labels 'y'
        as a list of tuples, '(x,y)'. This function will update the weights and
        biases of the network for the given training pairs with a specified
        rate'''
        
        update_b = [np.zeros(b.shape) for b in self.b];
        update_w = [np.zeros(w.shape) for w in self.w];
        
        for x,y in training_pairs:
            # print("trained pair")
            # print(x.shape,y.shape)
            self.feedForward(x);
            db, dw = self.backPropagate(y);
            
            update_b = [ub+ab for ub,ab in zip(update_b,db)];
            update_w = [uw+aw for uw,aw in zip(update_w,dw)];
        
        self.b = [b-(rate/len(training_pairs))*nb for b, nb in zip(self.b, update_b)];
        self.w = [w-(rate/len(training_pairs))*nw for w, nw in zip(self.w, update_w)];

    def randomTrainingProcedure(self, training_pairs, batch_size, eta, 
                                iterations, test_pairs=None):
        '''This randomizes the training procedure and allows for specification
        of batch size and iterations/epochs. This will more efficiently train
        the network.'''
        
        for i in range(iterations):
            print(f"Iteration - {i}");
            shuffle(training_pairs);
            
            for j in range(len(training_pairs)//batch_size):
                if j%1000==0: print(f"{j}/{len(training_pairs)//batch_size}");
                
                batch = training_pairs[j*batch_size:(j+1)*batch_size];
                self.trainNetwork(batch,1);
                
            if test_pairs != None:
                self.printAccuracy(test_pairs);
            
    def printAccuracy(self, training_pairs):
        '''This function prints the accuracy of the network for the training
        pairs provided consisting of input 'x' and labels 'y' as a list of 
        tuples, '(x,y)'.'''
        
        num_correct = 0;
        for ti,tl in training_pairs:        
            prediction = np.argmax(self.feedForward(ti));
            num_correct += (prediction == np.argmax(tl));
        
        print(f"{num_correct}/{len(training_pairs)} accuracy: \
              {(num_correct/len(training_pairs))*100}");

    @staticmethod
    def sigma(z):
        '''The activation function: sigmoidal curve.'''
        return( 1/(1 + np.exp(-z)) );

    def sigmaPrime(self,z):
        '''The derivative of the activation function'''
        return( self.sigma(z) * (1-self.sigma(z)) );
    
    def resetNetwork(self):
        #weights and biases
        self.w = [np.random.standard_normal(s)/s[1]**.5 for s in self.weight_shapes];
        self.b = [np.random.standard_normal((s,1)) for s in self.layer_sizes[:]];
    
    def setNetwork(self, w, b):
        #weights and biases
        self.w = w;
        self.b = b;
    
    def getNetwork(self):
        #weights and biases
        return(self.w,self.b);
    
    def getNodeWeights(self, l, i):
        '''Returns the outgoing weights of the node specified. This can be
        used for checking its pattern recognition.'''
        
        return ( self.w[l][i] );
    
    def getNodeActivation(self, l, i):
        '''Returns the activation of the node specified. This can be
        used for checking its pattern recognition.'''
        
        return (self.a[l][i] )
    
    def getActivations(self):
        '''Returns the activations of the whole network. This can be
        used for checking its pattern recognition.'''
        
        return (self.a)
    
    def setActivations(self,a):
        '''Returns the activations of the whole network. This can be
        used for checking its pattern recognition.'''
        
        self.a = a;
    
class CNN (NeuralNetwork) :
    
    def __init__(self, layer_sizes, image_dimension, kernel_size, kernel_number, maxpool_dimension):
        
        # print("given",maxpool_dimension);
        #make sure the maxpool_dimension doesn't use padding
        while image_dimension%maxpool_dimension != 0:
            maxpool_dimension -= 1;
        # print("use",maxpool_dimension);
        
        self.image_dimension = image_dimension;
        self.kernel_number = kernel_number;
        self.maxpool_size = image_dimension//maxpool_dimension;
        # print("block_size:",self.maxpool_size);
        
        #kernels of convolutional layers
        self.k = np.random.standard_normal(
            (kernel_number , kernel_size, kernel_size)) / kernel_size*kernel_size;
        
        layer_sizes = [maxpool_dimension*maxpool_dimension*kernel_number] + list(layer_sizes);
        
        super().__init__(layer_sizes);
        
    def convolveImage(self, image):
        '''Returns a list of the convolved images'''
        return [signal.convolve2d(image, self.k[i,:,:], mode = "same") for i in range(self.kernel_number)];
    
    def maxPoolImage(self, image):
        # print("image shape: ",image.shape)
        return block_reduce(image, block_size = (self.maxpool_size, self.maxpool_size), func = np.max);
    
    def convolution(self, image):
        '''Do one convolution and one maxpooling'''
        new_images = [];
        
        for i in range(self.kernel_number):
            new_image = self.maxPoolImage(signal.convolve2d(image, self.k[i,:,:], mode = "same"));
            new_images.append(new_image);
            # print("image:",new_image.shape);
            
        return np.array(new_images);
    
    def feedForward(self, x):
        '''Returns the output of output layer after given input to input layer.'''
        x = self.convolution(x.reshape((self.image_dimension,self.image_dimension)));
        x = x.reshape((self.layer_sizes[0],1));
        
        self.a[0] = x+self.b[0];
        
        for l in range(1,self.L):
            z_l = np.matmul(self.w[l-1],self.a[l-1]) + self.b[l];
            self.z[l] = z_l;
            self.a[l] = self.sigma(z_l);

        return(self.a[self.L-1]);
    
    def getNodeKernal(self, l, i):
        '''Returns the kernal of the specified node and layer.'''
        
        if l == 0:
            print("This is input layer, has no kernel.");
            return;
        if l == self.L-1:
            print("This is ouput layer, has no kernel.");
            return;
            
        return ( self.k[l][i] );
    
    def setLayerKernels(self, kind = "sobel"):
        if kind == "sobel":
            self.k = np.array([
                [[-1,0,1],[-2,0,2],[-1,0,1]],
                [[1,0,-1],[2,0,-2],[1,0,-1]],
                [[1,2,1],[0,0,0],[-1,-2,-1]],
                [[-1,-2,-1],[0,0,0],[1,2,1]]]
                );
        