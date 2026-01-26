#include <iostream>
#include "Value.hpp"
#include <set>
#include <Eigen/Dense>
#include <vector>


#include "NeuralNet.hpp"

using namespace std;

int main(){
    
    
    //std::shared_ptr<Value> a = std::make_shared<Value>(2.0);
    //a->label = "a";
    //std::shared_ptr<Value> b = std::make_shared<Value>(-3.0);   
    //b->label = "b";
    
    
    //std::shared_ptr<Value> e = (*a) * b;
    //e->gradient = 1;
    
    
    
    
    std::vector<int> layerShapes = {4, 4, 1};

    Mlp nn = Mlp(3, layerShapes);
    
    
    
    
    

    
    
    
    std::vector<std::vector<double>> xs = {{2, -3,-1}, {3, -1, 0.5}, {0.5, 1.0, 1.0}, {1, 1, -1}};
    

    

    std::vector<double> ys = {1, -1, -1, 1};
    

    double learningRate = 0.01;

    for (int i=0; i<100; i++){
        std::vector<std::shared_ptr<Value>> yPred;


        for (size_t i=0; i<xs.size();i++){
            yPred.push_back(nn.forward(xs[i]));
        }

        std::shared_ptr<Value> loss = make_shared<Value>(0);

        for (size_t i=0; i<ys.size();i++){
            loss = *((*((*yPred[i]) - ys[i]))^2) + loss;    
        }


        std::vector<std::shared_ptr<Value>> parameters = nn.parameters();
        
        
        for (auto& p: parameters){
            
            p->gradient = 0;

        }
        loss->applyBackWard();
        int counter = 0;
        for (auto& p: parameters){
            if (counter%41 == 0){
                std::cout<< "param gradient after prop: " << p->gradient<<endl;
            }
            counter+=1;
            
            //std::cout<< "param gradients: " << p->gradient<<endl;

        }
        
        for (auto& p: parameters){

            p->data = p->data - learningRate*p->gradient;
            if (counter%41 == 0){
                std::cout<< "param data: " << p->data<<endl;
            }
            //std::cout<< "param gradients: " << p->gradient<<endl;

        }

        
        double clip_value = 10.0;
        for (auto& p: parameters) {
            if (p->gradient > clip_value) p->gradient = clip_value;
            if (p->gradient < -clip_value) p->gradient = -clip_value;
        }

        std::cout<< "param ex: " << parameters[0]->data<<endl;

        std::cout<<"loss: "<<loss->data<<endl;



    }
    
    
    //loss->applyBackWard();
    
    
    


    
    

    



    //Value d = (e+c;
    // e.label = "e";
    // d.label = "d";

    // Value f = Value(-2);
    // f.label = "f";
    // Value L = d*f;
    // L.label = "L";

    // double h = 0.0000001;

    // std::cout<<"data "<<d.getData() << " label: " << d.label << " gradient: "<<d.grad <<endl;


    //derivative of tanh = 1 - tanh (x) ^ 2

}



