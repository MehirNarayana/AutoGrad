
#include "Value.hpp"
#include <cmath>


Value::Value(double data, std::set<std::shared_ptr<Value>> children, Op op):data(data), prev(children), op(op){
    
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other){
    std::shared_ptr<Value> out = std::make_shared<Value>(this->data + other->getData(), std::set<std::shared_ptr<Value>>{shared_from_this(), other}, Op::Add);
    auto currBackward = [=](){
        shared_from_this()->gradient += 1 * out->gradient ;
        other->gradient += 1 * out->gradient;

    };

    out->backward = currBackward;
    return out;



    
}


std::shared_ptr<Value> Value::operator+(double other){
    std::shared_ptr<Value> num = std::make_shared<Value>(other);
    std::shared_ptr<Value> out = std::make_shared<Value>(this->data + num->getData(), std::set<std::shared_ptr<Value>>{shared_from_this(), num}, Op::Add);
    auto currBackward = [=](){
        shared_from_this()->gradient += 1 * out->gradient ;
        num->gradient += 1 * out->gradient;

    };

    out->backward = currBackward;
    return out;



    
}

std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other){
    std::shared_ptr<Value> out = (*this) + (*(other)*-1);
    return out;
}


std::shared_ptr<Value> Value::operator-(double other){
    std::shared_ptr<Value> out = (*this) + ((other)*-1);
    return out;
}




std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other){
    std::shared_ptr<Value> out = std::make_shared<Value>(this->data * other->getData(), std::set<std::shared_ptr<Value>>{shared_from_this(), other}, Op::Mul);
    auto currBackward = [=](){
        shared_from_this()->gradient += other->data * out->gradient ;
        other->gradient += other->data * out->gradient;

    };

    out->backward = currBackward;
    return out;
}



std::shared_ptr<Value> Value::operator*(double other){
    std::shared_ptr<Value> num = std::make_shared<Value>(other);
    std::shared_ptr<Value> out = std::make_shared<Value>(this->data * num->getData(), std::set<std::shared_ptr<Value>>{shared_from_this(), num}, Op::Mul);
    auto currBackward = [=](){
        shared_from_this()->gradient += num->data * out->gradient ;
        num->gradient += num->data * out->gradient;

    };

    out->backward = currBackward;
    return out;



    
}

std::shared_ptr<Value> Value::operator^(double exponent){
    
    std::shared_ptr<Value> out = std::make_shared<Value>(std::pow(this->data, exponent), std::set<std::shared_ptr<Value>>{shared_from_this()}, Op::Pow);
    
    auto currBackward = [=](){
        shared_from_this()->gradient += exponent * std::pow(this->data, exponent - 1) * out->gradient ;
    };

    out->backward = currBackward;
    return out;
}



double Value::getData() const{
    return data;
}

std::string Value::getLabel(){
    return label;
}



std::shared_ptr<Value> Value::tanh(){
    double x = data;
    double t = std::tanh(x);
    std::shared_ptr<Value> out = std::make_shared<Value>(t, std::set<std::shared_ptr<Value>>{shared_from_this()});
    
    auto currBackward = [=](){
        shared_from_this()->gradient += (1 - std::pow(t, 2)) * out->gradient ;
        

    };

    out->backward = currBackward;
    
    out->label = "tanh";
    return out;
    
}



void Value::topoSort(std::shared_ptr<Value> root){
    if (visited.find(root)==visited.end()){
        visited.insert(root);
        for (const std::shared_ptr<Value>& parent: root->prev){
            topoSort(parent);
        }
        topo.push_back(root);
    }
    
}


void Value::applyBackWard(){
    visited.clear(); 
    topo.clear();
    topoSort(shared_from_this());
    gradient = 1;
    for (int i = topo.size() - 1; i >= 0; --i) {
        topo[i]->backward();
    
    }

}        







