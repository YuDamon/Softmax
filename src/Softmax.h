// V1.1


# ifndef _SoftmaxReg_H_
# define _SoftmaxReg_H_

# include <iostream>

const bool READ_FROM_FILE = 1;
const bool INSTANT = 0;

const bool SAVE_MODEL = 1;
const bool NOT_SAVE_MODEL = 0;

class SoftmaxReg {
public:
    SoftmaxReg(int _numClass, int _dim);
    ~SoftmaxReg();
    SoftmaxReg(const SoftmaxReg &softmaxReg);

    void Train(const std::string &filenameTrain, int epoch = 500, float alpha = 0.1, bool saveModel = 0);
    void Predict(const std::string &filenameTest, bool modelType = 0, const std::string &filenameModel = "");

protected:
	int getNum(const std::string &filename); 
    int loadData(const std::string &filename, int total, float **data, float **label, bool randPer);
	void LabelTrans(float temp, float *label);
	void CalcProb(float *x);

private:
	int numClass;    // num of categories
    int dim;        // dimension of features, bias not included
    float **theta; // weight
};

# endif