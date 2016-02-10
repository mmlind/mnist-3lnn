/**
 * @file 3lnn.c
 * @brief Neural network functionality for a 3-layer (INPUT, HIDDEN, OUTPUT) feed-forward, back-prop NN
 * @author Matt Lind
 * @date August 2015
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util/mnist-utils.h"
#include "3lnn.h"




/**
 * @details Retrieves a node via ID from a layer
 */

Node *getNode(Layer *l, int nodeId) {
    
    int nodeSize = sizeof(Node) + (l->nodes[0].wcount * sizeof(double));
    uint8_t *sbptr = (uint8_t*) l->nodes;
    
    sbptr += nodeId * nodeSize;
    
    return (Node*) sbptr;
}




/**
 * @brief Returns one of the layers of the network
 * @param nn A pointer to the NN
 * @param ltype Type of layer to be returned (INPUT, HIDDEN, OUTPUT)
 */

Layer *getLayer(Network *nn, LayerType ltype){
    
    Layer *l;
    
    switch (ltype) {
        case INPUT:{
            l = nn->layers;
            break;
        }
        case HIDDEN:{
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize;
            l = (Layer*)sbptr;
            break;
        }
            
        default:{ // OUTPUT
            uint8_t *sbptr = (uint8_t*) nn->layers;
            sbptr += nn->inpLayerSize + nn->hidLayerSize;
            l = (Layer*)sbptr;
            break;
        }
    }
    
    return l;
}




/**
 * @brief Returns the result of applying the given outputValue to the derivate of the activation function
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param outVal Output value that is to be back propagated
 */

double getActFctDerivative(Network *nn, LayerType ltype, double outVal){
    
    double dVal = 0;
    ActFctType actFct;
    
    if (ltype==HIDDEN) actFct = nn->hidLayerActType;
                  else actFct = nn->outLayerActType;
    
    if (actFct==TANH) dVal = 1-pow(tanh(outVal),2);
                 else dVal = outVal * (1-outVal);
    
    return dVal;
}




/**
 * @brief Updates a node's weights based on given error
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param id Sequential id of the node that is to be calculated
 * @param error The error (difference between desired output and actual output
 */

void updateNodeWeights(Network *nn, LayerType ltype, int id, double error){
    
    Layer *updateLayer = getLayer(nn, ltype);
    Node *updateNode = getNode(updateLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    if (ltype==HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    } else {
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    for (int i=0; i<updateNode->wcount; i++){
        Node *prevLayerNode = (Node*)sbptr;
        updateNode->weights[i] += (nn->learningRate * prevLayerNode->output * error);
        sbptr += prevLayerNodeSize;
    }
    
    // update bias weight
    updateNode->bias += (nn->learningRate * 1 * error);
    
}




/**
 * @brief Back propagates network error to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */

void backPropagateHiddenLayer(Network *nn, int targetClassification){
    
    Layer *ol = getLayer(nn, OUTPUT);
    Layer *hl = getLayer(nn, HIDDEN);
    
    for (int h=0;h<hl->ncount;h++){
        Node *hn = getNode(hl,h);
        
        double outputcellerrorsum = 0;
        
        for (int o=0;o<ol->ncount;o++){
            
            Node *on = getNode(ol,o);
            
            int targetOutput = (o==targetClassification)?1:0;
            
            double errorDelta = targetOutput - on->output;
            double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
            
            outputcellerrorsum += errorSignal * on->weights[h];
        }
        
        double hiddenErrorSignal = outputcellerrorsum * getActFctDerivative(nn, HIDDEN, hn->output);
        
        updateNodeWeights(nn, HIDDEN, h, hiddenErrorSignal);
    }
    
}




/**
 * @brief Back propagates network error in output layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */

void backPropagateOutputLayer(Network *nn, int targetClassification){
    
    Layer *ol = getLayer(nn, OUTPUT);
    
    for (int o=0;o<ol->ncount;o++){
        
        Node *on = getNode(ol,o);
        
        int targetOutput = (o==targetClassification)?1:0;
        
        double errorDelta = targetOutput - on->output;
        double errorSignal = errorDelta * getActFctDerivative(nn, OUTPUT, on->output);
        
        updateNodeWeights(nn, OUTPUT, o, errorSignal);
        
    }
    
}




/**
 * @brief Back propagates network error from output layer to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */

void backPropagateNetwork(Network *nn, int targetClassification){
    
    backPropagateOutputLayer(nn, targetClassification);
    
    backPropagateHiddenLayer(nn, targetClassification);
    
}




/**
 * @brief Performs an activiation function (as defined in the NN's defaults) to a specified node
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param id Sequential id of the node that is to be calculated
 */

void activateNode(Network *nn, LayerType ltype, int id){
    
    Layer *l = getLayer(nn, ltype);
    Node *n = getNode(l, id);
    
    ActFctType actFct;
    
    if (ltype==HIDDEN) actFct = nn->hidLayerActType;
    else actFct = nn->outLayerActType;
    
    if (actFct==TANH)   n->output = tanh(n->output);
    else n->output = 1 / (1 + (exp((double)-n->output)) );
    
}




/**
 * @brief Calculates the output value of a specified node by multiplying all its weights with the previous layer's outputs
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 * @param id Sequential id of the node that is to be calculated
 */

void calcNodeOutput(Network *nn, LayerType ltype, int id){
    
    Layer *calcLayer = getLayer(nn, ltype);
    Node *calcNode = getNode(calcLayer, id);
    
    Layer *prevLayer;
    int prevLayerNodeSize = 0;
    
    if (ltype==HIDDEN) {
        prevLayer = getLayer(nn, INPUT);
        prevLayerNodeSize = nn->inpNodeSize;
    }
    else {
        prevLayer = getLayer(nn, HIDDEN);
        prevLayerNodeSize = nn->hidNodeSize;
    }
    
    uint8_t *sbptr = (uint8_t*) prevLayer->nodes;
    
    // Start by adding the bias
    calcNode->output = calcNode->bias;

    
    for (int i=0; i<prevLayer->ncount;i++){
        Node *prevLayerNode = (Node*)sbptr;
        calcNode->output += prevLayerNode->output * calcNode->weights[i];
        sbptr += prevLayerNodeSize;
    }

}




/**
 * @brief Calculates the output values of a given NN layer
 * @param nn A pointer to the NN
 * @param ltype Type of layer (INPUT, HIDDEN, OUTPUT)
 */

void calcLayer(Network *nn, LayerType ltype){
    Layer *l;
    l = getLayer(nn, ltype);
    
    for (int i=0;i<l->ncount;i++){
        calcNodeOutput(nn, ltype, i);
        activateNode(nn,ltype,i);
    }
}




/**
 * @brief Feeds input layer values forward to hidden to output layer (calculation and activation fct)
 * @param nn A pointer to the NN
 */

void feedForwardNetwork(Network *nn){
    calcLayer(nn, HIDDEN);
    calcLayer(nn, OUTPUT);
}




/**
 * @brief Feeds some Vector data into the INPUT layer of the NN
 * @param nn A pointer to the NN
 * @param v A pointer to a vector
 */

void feedInput(Network *nn, Vector *v) {
    
    Layer *il;
    il = nn->layers;
    
    Node *iln;
    iln = il->nodes;
    
    // Copy the vector content to the "output" field of the input layer nodes
    for (int i=0; i<v->size;i++){
        iln->output = v->vals[i];
        iln++;               // @warning This only works because inputNodeSize = sizeof(Node)
    }
    
}




/**
 * @details Creates an input layer and sets all weights to random values [0-1]
 * @param inpCount Number of nodes in the input layer
 */

Layer *createInputLayer(int inpCount){
    
    int inpNodeSize     = sizeof(Node);         // Input layer has 0 weights
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    Layer *il = malloc(inpLayerSize);
    il->ncount = inpCount;
    
    // Create a detault input layer node
    Node iln;
    iln.bias = 0;
    iln.output = 0;
    iln.wcount = 0;
    
    // Use a single byte pointer to fill in the network's content
    uint8_t *sbptr = (uint8_t*) il->nodes;
    
    // Copy the default input layer node x times
    for (int i=0;i<il->ncount;i++){
        memcpy(sbptr,&iln,inpNodeSize);
        sbptr += inpNodeSize;
    }
    
    return il;
}



/**
 * @details Creates a layer and sets all weights to random values [0-1]
 * @param nodeCount Number of nodes
 * @param weightCount Number of weights per node
 */

Layer *createLayer(int nodeCount, int weightCount){
    
    int nodeSize = sizeof(Node) + (weightCount * sizeof(double));
    Layer *l = (Layer*)malloc(sizeof(Layer) + (nodeCount*nodeSize));
    
    l->ncount = nodeCount;
    
    // create a detault node
    Node *dn = (Node*)malloc(sizeof(Node) + ((weightCount)*sizeof(double)));
    dn->bias = 0;
    dn->output = 0;
    dn->wcount = weightCount;
    for (int o=0;o<weightCount;o++) dn->weights[o] = 0; // will be initialized later
    
    uint8_t *sbptr = (uint8_t*) l->nodes;     // single byte pointer
    
    // copy the default cell to all cell positions in the layer
    for (int i=0;i<nodeCount;i++) memcpy(sbptr+(i*nodeSize),dn,nodeSize);
    
    free(dn);
    
    return l;
}




/**
 * @brief Initializes the NN by creating and copying INTPUT, HIDDEN, OUTPUT data structures into the NN's memory space
 * @param nn A pointer to the NN
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */

void initNetwork(Network *nn, int inpCount, int hidCount, int outCount){
    
    // Copy the input layer into the network's memory block and delete it
    Layer *il = createInputLayer(inpCount);
    memcpy(nn->layers,il,nn->inpLayerSize);
    free(il);
    
    // Move pointer to end of input layer = beginning of hidden layer
    uint8_t *sbptr = (uint8_t*) nn->layers;     // single byte pointer
    sbptr += nn->inpLayerSize;
    
    // Copy the hidden layer into the network's memory block and delete it
    Layer *hl = createLayer(hidCount, inpCount);
    memcpy(sbptr,hl,nn->hidLayerSize);
    free(hl);
    
    // Move pointer to end of hidden layer = beginning of output layer
    sbptr += nn->hidLayerSize;
    
    // Copy the output layer into the network's memory block and delete it
    Layer *ol = createLayer(outCount, hidCount);
    memcpy(sbptr,ol,nn->outLayerSize);
    free(ol);
    
}




/**
 * @brief Sets the default network parameters (which can be overwritten/changed)
 * @param nn A pointer to the NN
 */

void setNetworkDefaults(Network *nn){
    
    // Set deffault activation function types
    nn->hidLayerActType = SIGMOID;
    nn->outLayerActType = SIGMOID;
    
    nn->learningRate    = 0.004;    // TANH 78.0%
    nn->learningRate    = 0.2;      // SIGMOID 91.5%
    
}




/**
 * @brief Initializes a layer's weights with random values
 * @param nn A pointer to the NN
 * @param ltype Defining what layer to initialize
 */

void initWeights(Network *nn, LayerType ltype){
    
    int nodeSize = 0;
    if (ltype==HIDDEN) nodeSize=nn->hidNodeSize;
                  else nodeSize=nn->outNodeSize;
    
    Layer *l = getLayer(nn, ltype);
    
    uint8_t *sbptr = (uint8_t*) l->nodes;
    
    for (int o=0; o<l->ncount;o++){
    
        Node *n = (Node *)sbptr;
        
        for (int i=0; i<n->wcount; i++){
            n->weights[i] = 0.7*(rand()/(double)(RAND_MAX));
            if (i%2) n->weights[i] = -n->weights[i];  // make half of the weights negative
        }
        
        // init bias weight
        n->bias =  rand()/(double)(RAND_MAX);
        if (o%2) n->bias = -n->bias;  // make half of the bias weights negative
        
        sbptr += nodeSize;
    }
    
}



/**
 * @brief Creates a dynamically-sized, 3-layer (INTPUT, HIDDEN, OUTPUT) neural network
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */

Network *createNetwork(int inpCount, int hidCount, int outCount){
    
    // Calculate size of INPUT Layer
    int inpNodeSize     = sizeof(Node);         // Input layer has 0 weights
    int inpLayerSize    = sizeof(Layer) + (inpCount * inpNodeSize);
    
    // Calculate size of HIDDEN Layer
    int hidWeightsCount = inpCount;
    int hidNodeSize     = sizeof(Node) + (hidWeightsCount * sizeof(double));
    int hidLayerSize    = sizeof(Layer) + (hidCount * hidNodeSize);
    
    // Calculate size of OUTPUT Layer
    int outWeightsCount = hidCount;
    int outNodeSize     = sizeof(Node) + (outWeightsCount * sizeof(double));
    int outLayerSize    = sizeof(Layer) + (outCount * outNodeSize);
    
    // Allocate memory block for the network
    Network *nn = (Network*)malloc(sizeof(Network) + inpLayerSize + hidLayerSize + outLayerSize);
    
    // Set/remember byte sizes of each component of the network
    nn->inpNodeSize     = inpNodeSize;
    nn->inpLayerSize    = inpLayerSize;
    nn->hidNodeSize     = hidNodeSize;
    nn->hidLayerSize    = hidLayerSize;
    nn->outNodeSize     = outNodeSize;
    nn->outLayerSize    = outLayerSize;
    
    // Initialize the network by creating the INPUT, HIDDEN and OUTPUT layer inside of it
    initNetwork(nn, inpCount, hidCount, outCount);
    
    // Setting defaults
    setNetworkDefaults(nn);
    
    // Init connection weights with random values
    initWeights(nn, HIDDEN);
    initWeights(nn, OUTPUT);
    
    return nn;
}




/**
 * @brief Returns the network's classification using the ID of teh node with the hightest output
 * @param nn A pointer to the NN
 */

int getNetworkClassification(Network *nn){
    
    Layer *l = getLayer(nn, OUTPUT);
    
    double maxOut = 0;
    int maxInd = 0;
    
    for (int i=0; i<l->ncount; i++){
        
        Node *on = getNode(l,i);

        if (on->output > maxOut){
            maxOut = on->output;
            maxInd = i;
        }
    }
    
    return maxInd;
}






/**
 * @brief DEBUGGING function
 * @details Prints all the weights (either pointers' target addresses or actual values) on the screen for debugging purposes
 */

void displayNetworkWeightsForDebugging(Network *nn){
    
    // only print the first x and last x nodes/connections (to improve legible rendering in the console screen)
    int topLast = 6;
    
    for (int l=1; l<2;l++){
        
        Layer *layer = getLayer(nn, OUTPUT);
        
        printf("Layer %d   Weights\n\n",l);
        
        // print connections per node and WEIGHT of connection
        
        if (layer->ncount>0){
            
            printf("Layer %d   NodeId  |  ConnectionId:WeightAddress \n\n",l);
            
            int kSize = 5*5;
            
            // table header
            printf("Node | ");
            for (int x=0; x<kSize; x++) {if (x<topLast || x>=kSize-topLast) printf(" conn:address  ");}
            printf("\n-------");
            for (int x=0; x<kSize; x++) {if (x<topLast || x>=kSize-topLast) printf("---------------");}
            printf("\n");
            
            for (int n=0; n<10; n++){
                
                printf("%4d | ",n);
                
                Node *node = getNode(layer, n);
                
                int connCount = node->wcount;
                
                for (int c=0; c<connCount; c++){
                    
                    // Dereference the weightPointer to validate its pointing to a valid weight
                    double w = node->weights[c];
                    
                    if (c<topLast || c>=connCount-topLast) printf("%5d:%9f",c,w);
                }
                printf("\n");
                
                
            }
            printf("\n\n");
            
            
        }
        
    }
    
}

