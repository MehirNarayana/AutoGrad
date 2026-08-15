
#pragma once
#include <iostream>
#include <set>
#include <memory>

enum class Op {
        Add,
        Mul,
        Sub,
        Div,
        Pow,
        None
};


class Value: public std::enable_shared_from_this<Value>{
    public:
        Value(double data, std::set<std::shared_ptr<Value>> children = {}, Op op = Op::None);
        double getData() const;
        std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
        std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);
        std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& other);
        std::shared_ptr<Value> operator*(double other);
        std::shared_ptr<Value> operator+(double other);
        std::shared_ptr<Value> operator^(double exponent);
        std::shared_ptr<Value> operator-(double other);
        void topoSort(std::shared_ptr<Value> root);
        void applyBackWard();

        Op op;

        std::vector<std::shared_ptr<Value>> topo;
        std::set<std::shared_ptr<Value>> visited;
        std::string getLabel();
        double data;
        std::set<std::shared_ptr<Value>> prev;
        std::string label = "";
        double  gradient = 0.0;
        std::function<void()> backward = [](){};
        std::shared_ptr<Value> tanh();









};


