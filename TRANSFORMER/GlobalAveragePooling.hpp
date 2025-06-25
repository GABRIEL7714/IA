#pragma once
#include <vector>

using namespace std;

class GlobalAveragePooling {
public:
    vector<float> forward(const vector<vector<float>>& tokens) const;
};
