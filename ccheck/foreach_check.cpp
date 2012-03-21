#include<iostream>
#include<vector>

int main(int argc,char **arv){
    std::vector<int> vec = {1,2,3,4};
    for(int &v: vec){
       std::cout<<v<<std::endl; 
    }
    return 0;
}
