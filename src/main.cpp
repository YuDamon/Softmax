 # include <iostream>
 # include <string>

 # include "SoftmaxReg.h"

using namespace std;

 int main() {
 	const string fileTrain = "irisTrain.txt";
 	const string fileTest = "IrisTest.txt";
	const int dim = 4;				//dimension of features (bias not included)
	const int numClass = 3;			//number of categories
	const int epoch = 400;			//Iteration
	const float alpha = 0.25;		// Learning Rate

 	SoftmaxReg Classifier(numClass, dim);
 	Classifier.Train(fileTrain, epoch, alpha, SAVE_MODEL);
 	Classifier.Predict(fileTest);

 	return 0;
 }
