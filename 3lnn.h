/**
 * @file 3lnn.h
 * @brief Neural network functionality for a 3-layer (INPUT, HIDDEN, OUTPUT) feed-forward, back-prop NN
 * @author Matt Lind
 * @date August 2015
 */





typedef struct Network Network;
typedef struct Layer Layer;
typedef struct Node Node;
typedef struct Vector Vector;

typedef enum LayerType {INPUT, HIDDEN, OUTPUT} LayerType;
typedef enum ActFctType {SIGMOID, TANH} ActFctType;




/**
 * @brief Dynamic data structure containing defined number of values
 */

struct Vector{
    int size;
    double vals[];
};




/**
 * @brief Dynamic data structure modeling a neuron with a variable number of connections/weights
 */

struct Node{
    double bias;
    double output;
    int wcount;
    double weights[];
};




/**
 * @brief Dynamic data structure holding a definable number of nodes that form a layer
 */

struct Layer{
    int ncount;
    Node nodes[];
};


/**
 * @brief Dynamic data structure holding the whole network
 */

struct Network{
    int inpNodeSize;
    int inpLayerSize;
    int hidNodeSize;
    int hidLayerSize;
    int outNodeSize;
    int outLayerSize;
    double learningRate;         ///< Factor by which connection weight changes are applied
    ActFctType hidLayerActType;
    ActFctType outLayerActType;
    Layer layers[];
};




/**
 * @brief Creates a dynamically-sized, 3-layer (INTPUT, HIDDEN, OUTPUT) neural network
 * @param inpCount Number of nodes in the INPUT layer
 * @param hidCount Number of nodes in the HIDDEN layer
 * @param outCount Number of nodes in the OUTPUT layer
 */

Network *createNetwork(int inpCount, int hidCount, int outCount);




/**
 * @brief Feeds some Vector data into the INPUT layer of the NN
 * @param nn A pointer to the NN
 * @param v A pointer to a vector
 */

void feedInput(Network *nn, Vector *v);




/**
 * @brief Feeds input layer values forward to hidden to output layer (calculation and activation fct)
 * @param nn A pointer to the NN
 */

void feedForwardNetwork(Network *nn);




/**
 * @brief Back propagates network error from output layer to hidden layer
 * @param nn A pointer to the NN
 * @param targetClassification Correct classification (=label) of the input stream
 */

void backPropagateNetwork(Network *nn, int targetClassification);




/**
 * @brief Returns the network's classification using the ID of teh node with the hightest output
 * @param nn A pointer to the NN
 */

int getNetworkClassification(Network *nn);




void displayNetworkWeightsForDebugging(Network *nn);

