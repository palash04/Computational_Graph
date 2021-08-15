#include <bits/stdc++.h>
using namespace std;

struct Node{
    double val, gradient;
    string type, op;
    vector<Node*> children, parents;
    int indegree, outdegree;
    Node() {
        val = -1;       // not assigned yet
        type = "none";  // no type assigned yet  (type: input_node, weight_node, output_node, other)
        op = "none";    // no operator assigned yet
        indegree = 0;
        outdegree = 0;
        gradient = 0.0;
    }
};

vector<double> init_weights(int numOfWeightNodes) {
    vector<double> weights;
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(0,1);
    for (int i=0; i<numOfWeightNodes; ++i) {
        double number = distribution(generator);
        weights.push_back(number);
    }

    return weights;
}

class Graph{
    vector<Node*> iso_nodes;    // isolated nodes like input_nodes, weight_nodes
    vector<Node*> output_nodes;
    vector<Node*> input_nodes;
    vector<Node*> topoOrder;
    vector<Node*> topoOrderReversed;
    vector<Node*> softOutput_nodes;
    vector<Node*> preSoftOutput_nodes;
    int label;
public:
    void linear(int numOfInputNodes, int numOfOutputNodes, bool first_layer) {

        numOfInputNodes += 1;       // for bias

        int numOfWeightNodes = numOfInputNodes * numOfOutputNodes;
        vector<Node*> input_vec(numOfInputNodes);
        vector<Node*> weight_vec(numOfWeightNodes);
        vector<Node*> hidden_vec(numOfOutputNodes);

        // weights init
        vector<double> weigths = init_weights(numOfWeightNodes);

        if (first_layer) {
            // creating input nodes
            for (int i=0; i < numOfInputNodes; i++) {
                Node *newNode = new Node();
                newNode->type = "input_node";
                newNode->indegree = 0;
                input_vec[i] = newNode;
                if (i == numOfInputNodes - 1) {
                    newNode->val = 1.0;
                }
            }
            iso_nodes = input_vec;
            input_nodes = input_vec;
        }else {
            input_vec = output_nodes;
            Node *newNode = new Node();
            newNode->type = "input_node";
            newNode->indegree = 0;
            newNode->val = 1.0;
            input_vec.push_back(newNode);
            iso_nodes.push_back(newNode);
        }

        // creating weight nodes
        for (int i=0;i<numOfWeightNodes;i++) {
            Node *newNode = new Node();
            newNode->type = "weight_node";
            newNode->val = weigths[i];
            newNode->indegree = 0;
            weight_vec[i] = newNode;
        }

        // creating hidden/output nodes
        for (int i=0; i < numOfOutputNodes; i++) {
            Node *newNode = new Node();
            newNode->type = "hidden_node";
            hidden_vec[i] = newNode;
        }

        // creating operator (*) nodes for (W^T * X)
        vector<Node*> operator_vec(numOfWeightNodes);
        int count = 0;
        for (int i=0;i<numOfWeightNodes;i++) {
            Node *newNode = new Node();
            newNode->type = "operator_node";
            newNode->op = "multiply";
            Node *xi = input_vec[count];        // input x (scalar)
            Node *wi = weight_vec[i];           // weight w (scalar)

            newNode->parents.push_back(xi);
            newNode->parents.push_back(wi);
            newNode->indegree = 2;

            xi->children.push_back(newNode);
            xi->outdegree += 1;
            wi->children.push_back(newNode);
            wi->outdegree += 1;

            count = (count + 1) % numOfInputNodes;
            operator_vec[i] = newNode;
        }

        // connecting operator nodes with hidden nodes
        int s = 0;
        int e = s + numOfInputNodes - 1;
        for (int i=0;i<numOfOutputNodes;i++) {

            int X = 0;
            int Y = 1;
            auto start = operator_vec.begin() + s;
            auto end = operator_vec.begin() + e + 1;

            vector<Node*> operator_vec_hi(e - s + 1);

            copy(start, end, operator_vec_hi.begin());

            // connect them
            Node *hi = hidden_vec[i];
            hi->op = "add";

            for (int j=0;j<numOfInputNodes;j++) {
                Node *operator_hi = operator_vec_hi[j];
                hi->parents.push_back(operator_hi);
                operator_hi->children.push_back(hi);
                hi->indegree += 1;
                operator_hi->outdegree += 1;
            }
            s = e + 1;
            e = s + numOfInputNodes - 1;
        }

        iso_nodes.insert(iso_nodes.end(), weight_vec.begin(), weight_vec.end());

        output_nodes.clear();
        output_nodes = hidden_vec;
    }

    void softmax() {
        int n_outputs = (int)output_nodes.size();

        vector<Node*> soft_outputs(n_outputs);

        for (int i=0;i<n_outputs;i++) {
            Node *outputi = output_nodes[i];

            Node *newNode = new Node();
            newNode->type = "output_node";
            newNode->op = "softmax";

            newNode->parents.push_back(outputi);
            outputi->children.push_back(newNode);
            newNode->indegree += 1;
            outputi->outdegree += 1;
            soft_outputs[i] = newNode;
        }
        preSoftOutput_nodes = output_nodes;
        softOutput_nodes = soft_outputs;
    }

    void sigmoid() {
        int n_outputs = (int)output_nodes.size();

        vector<Node*> sig_outputs(n_outputs);

        for (int i=0;i<n_outputs;i++) {
            Node *outputi = output_nodes[i];

            Node *newNode = new Node();
            newNode->type = "output_node";
            newNode->op = "sigmoid";

            newNode->parents.push_back(outputi);
            outputi->children.push_back(newNode);
            newNode->indegree += 1;
            outputi->outdegree += 1;
            sig_outputs[i] = newNode;
        }
        output_nodes = sig_outputs;
    }

    void topological_ordering(bool reverse_topology) {
        vector<Node*> topologicalOrder;
        queue<Node*> q;

        if (!reverse_topology) {
            for (int i=0;i<iso_nodes.size();i++) {
                q.push(iso_nodes[i]);
            }

            while (!q.empty()) {
                Node *node = q.front();q.pop();
                topologicalOrder.push_back(node);
                for (auto child : node->children) {
                    child->indegree -= 1;
                    if (child->indegree == 0) q.push(child);
                }
            }
            topoOrder = topologicalOrder;
        }else {
            for (int i=0;i<output_nodes.size();i++) {
                q.push(output_nodes[i]);
            }

            while (!q.empty()) {
                Node *node = q.front();q.pop();
                topologicalOrder.push_back(node);
                for (auto parent : node->parents) {
                    parent->outdegree -= 1;
                    if (parent->outdegree == 0) q.push(parent);
                }
            }
            topoOrderReversed = topologicalOrder;
        }
    }

    vector<Node*> forward(vector<double> &X) {
        for (int i=0;i<X.size();i++) {
            input_nodes[i]->val = X[i];
        }

        int n = (int)topoOrder.size();
        for (int t=0;t<n;t++) {
            Node *node = topoOrder[t];
            vector<Node*> parents = node->parents;
            double node_val = 1;

            if (node->op == "multiply") {
                for (auto parent:parents) {
                    node_val *= parent->val;
                }
                node->val = node_val;
            }else if (node->op == "add") {
                node_val = 0;
                for (auto parent:parents) {
                    node_val += parent->val;
                }
                node->val = node_val;
            }else if (node->op == "softmax") {
                double denominator = 0.0;
                int n_outputs = (int)preSoftOutput_nodes.size();

                for (int i=0;i<n_outputs;i++) {
                    denominator += exp(preSoftOutput_nodes[i]->val);
                }
                assert (parents.size() == 1);
                double p_output = parents[0]->val;
                double numerator = exp(p_output);
                node->val = numerator / denominator;
            }else if (node->op == "sigmoid") {
                assert (parents.size() == 1);
                double p_output = parents[0]->val;
                node->val = 1 / (1 + exp(-p_output));
            }
        }
        return softOutput_nodes;
    }

    void CrossEntropyLoss() {
        Node *newNode = new Node();
        newNode->type = "loss_node";
        newNode->op = "cross_entropy_loss";
        newNode->gradient = 1.0;
        for (auto parent : softOutput_nodes) {
            newNode->parents.push_back(parent);
            parent->children.push_back(newNode);
            newNode->indegree += 1;
            parent->outdegree += 1;
        }

        output_nodes.clear();
        output_nodes.push_back(newNode);
    }

    double criterion(vector<Node*> &output, int target) {
        label = target;
        double loss = 0.0;
        for (int i=0;i<output.size();i++) {
            if (target == i) {
                double pi = output[i]->val;
                loss = -1 * log(pi);
            }
        }
        return loss;
    }

    void backward() {
        double alpha = 1e-3; // learning rate

        for (int t=0;t<topoOrderReversed.size();t++) {
            Node *node = topoOrderReversed[t];
            for (int i=0;i<node->parents.size();i++) {
                if (node->op == "add") {
                    node->parents[i]->gradient +=  (node->gradient * 1.0);
                }else if (node->op == "multiply") {
                    Node *par1 = node->parents[0];
                    Node *par2 = node->parents[1];
                    if (par1 == node->parents[i]) {
                        node->parents[i]->gradient += (node->gradient * par2->val);
                    }else {
                        node->parents[i]->gradient += (node->gradient * par1->val);
                    }
                }else if (node->op == "cross_entropy_loss") {
                    double den = node->parents[i]->val;
                    double grad = 0.0;
                    if (label == i) {
                        grad = -1.0 / den;
                    }
                    node->parents[i]->gradient += (node->gradient * grad);
                }else if (node->op == "softmax") {
                    double soft_outputi = node->val;
                    double grad = soft_outputi * (1.0 - soft_outputi);
                    node->parents[i]->gradient += (node->gradient * grad);
                }else if (node->op == "sigmoid") {
                    double sig_outputi = node->val;
                    double grad = sig_outputi * (1.0 - sig_outputi);
                    node->parents[i]->gradient += (node->gradient * grad);
                }
            }
            // update weight
            if (node->type == "weight_node") {
                node->val = node->val - alpha * node->gradient;
            }
        }
    }
};

// normalize input vector
void normalize(vector<double> &X) {
    int mx = *max_element(X.begin(), X.end());
    for (int i=0;i<X.size();i++) {
        X[i] = X[i] / mx;
    }
}

int main() {
    Graph g;

    // number of layers
    int n = 3;

    vector<int> layers_info(n);
    layers_info = {3,4,10};

    vector<double> X = {120,224,12};
    int target = 3;

    normalize(X);

    vector<Node*> last_layer;
    for (int i=0;i<n-1;i++) {
        if (i == 0)
            g.linear(layers_info[i],layers_info[i+1], true);
        else
            g.linear(layers_info[i],layers_info[i+1], false);

        if (i != n-2) {
            g.sigmoid();
        }
    }

    // add softmax layer
    g.softmax();
    g.CrossEntropyLoss();

    g.topological_ordering(false);
    g.topological_ordering(true);

    int epochs = 10;

    for (int epoch=1;epoch<=epochs;epoch++) {
        cout << "Epoch : " << epoch;
        // Forward pass
        vector<Node*> output = g.forward(X);

        // Calculate Loss
        double loss = g.criterion(output, target);
        cout << "\tLoss: " << loss;

        // Update Weights
        g.backward();

        double pred_value = INT_MIN;
        int pred_label;

        for (int i=0;i<output.size();i++) {
            if (output[i]->val > pred_value) {
                pred_label = i;
                pred_value = output[i]->val;
            }
        }
        cout << "\tPred: " << pred_label << "\n";
    }
}