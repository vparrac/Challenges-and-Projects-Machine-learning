import numpy as np
import pdb

def train(neuronalNetwork, X, y, cost, costDev, devActivation, learningRate=0.05, train=True ):
    """This method train the neuronal nerworks pass by parameter
    Parameters
    neuronalNetowrk: The neuronal Network
    X: A list[M,n] where
        M is the among of data
        n is the among of attibutes of each data
    y: list[1,M] with the real class of each attribute
    cost: Fuction: the cost function of the net
    learningRate: Number The learning rate of the net
    train: Boolean if it is false, the net predict the class for the imput, 
    if it is True, the methon train the net
    """
    out = [[X[:,np.newaxis],None]]    
       

    for layer in neuronalNetwork:      
        #pdb.set_trace()                     
        z = out[-1][0].T@layer.weigths+layer.bias        
        za = layer.activationFunction(z)        
        out.append([za,z[0]])            
    if train: 
        deltas=[]          
        for i in reversed(range(len(neuronalNetwork))):  
            #print(i)
            
            if i == len(neuronalNetwork)-1:                
                #dC/dW_L
                dC= costDev(out[i+1][0],y)                
                dFa= devActivation(out[i+1][0])
                dk= dC@dFa.T
                deltas.append([dk])                                     
            else:

                #dC/dW_L                        
                dk=deltas[-1] #Esto es igual a dL
                W= neuronalNetwork[i+1].weigths
                dFa= devActivation(out[i+2][0])                
                dW=[]                
                dFa=dFa[:,np.newaxis]    
                if len(dk)==1:                       
                    dWi=(dFa[0]*W)*dk[0]   
                    dW=dWi[0]       
                             
                else: 
                    dp=np.array(dk)
                    dp=dp[:,np.newaxis]   
                    dW= dFa.T*(dp.T@W)
                    dW=dW[0]
                
                if(np.isscalar(dW)):
                    deltas.append([dW])
                else:
                    deltas.append(dW)
                
            if len(deltas[-1])==1:       
                neuronalNetwork[i].weigths=neuronalNetwork[i].weigths-learningRate*(deltas[-1][0]*out[i][0])
            elif len(out[i][0])==1:
                neuronalNetwork[i].weigths=neuronalNetwork[i].weigths-learningRate*(deltas[-1]*out[i][0][0])
            else: 
                delk=deltas[-1]
                zetta= out[i][0]
                dev=np.zeros((len(zetta),len(delk)))
                for p in range(len(zetta)):                    
                    for q in range(len(delk)):
                        #pdb.set_trace()
                        dev[p,q]=zetta[p]*delk[q]
        
                neuronalNetwork[i].weigths=neuronalNetwork[i].weigths-learningRate*dev
            
    return out[-1][0]



