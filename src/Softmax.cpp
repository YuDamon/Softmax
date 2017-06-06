# include <iostream>
# include <cmath>
# include <fstream>
# include <string>
# include <time.h>

# include "SoftmaxReg.h"

using namespace std;

SoftmaxReg::SoftmaxReg (int _numClass, int _dim):numClass(_numClass), dim(_dim) {
	
	// initialize theta
	theta = new float*[numClass];
	for (int i = 0; i < numClass; i++) 
		theta[i] = new float[dim + 1];
	for (int i = 0; i < dim + 1; i++) 
		for (int j = 0; j < numClass; j++) 
			theta[j][i] = 0.01;
}


SoftmaxReg::~SoftmaxReg() { 
	for(int i = 0; i < numClass; i++) 
		delete[] theta[i];
	delete[] theta;
}

int SoftmaxReg::getNum(const string &filename) { 
	fstream fin;
	fin.open(filename,ios::in);
	if (fin.fail()) {
		cerr << "Error: get num failed" <<endl;
		return 0;
	}
	string line;
	int count = 0;
	while (!fin.eof()) {
		getline(fin,line);
		count++; 
	}
	fin.close();
	return count;
}

int SoftmaxReg::loadData(const string &filename, int total, float **data, float **label, bool randPer) {
	fstream fin;
	fin.open(filename, ios::in);
	if (fin.fail()) {
		cerr << "Function loadData() failed" <<endl;
		return 0;
	}
	int *rIdx = new int[total];
	for(int i = 0; i < total; i++) 
		rIdx[i] = i;

	if (randPer == true) {
		int rNum = 0;
		int tmp = 0;
		srand((int)time(NULL));
		for(int i = 0; i < total; i++) {
			rNum = rand()%total;
			tmp = rIdx[i];
			rIdx[i] = rIdx[rNum];
			rIdx[rNum] = tmp;
		}
	}

	float tempLabel = -1; 
	for (int i = 0; i < total; i++) {
		data[rIdx[i]][0] = 1;
		for (int j = 1; j < dim+1; j++) {
			fin >> data[rIdx[i]][j];
		}	
		fin >> tempLabel;
		LabelTrans(tempLabel, label[rIdx[i]]);
	} 
	delete[] rIdx;
	return 1;
}
void SoftmaxReg::LabelTrans (float temp, float *label) { 
	for (int i = 0; i < numClass; i++) {
		label[i] = 0;
	}
	int t = (int)(temp+0.1);
	label[t] = 1;
}

void SoftmaxReg::Train(const string &filenameTrain, int epoch, float alpha, bool saveModel) {
	
	float **dataTrain, **labelTrain;
	int numTrain = getNum(filenameTrain); 
	dataTrain = new float*[numTrain];
	labelTrain = new float*[numTrain];

	for (int i = 0; i < numTrain; i++) {
		dataTrain[i] = new float[dim + 1];
		labelTrain[i] = new float[numClass];
	}
	cout << "Training INFO -------------"<<endl;
	//load data
	if (loadData(filenameTrain, numTrain, dataTrain, labelTrain, true) == 0) {
		cerr << "INFO    [Error]: Training failed" << endl;
		return;
	}

	float *prob = new float[numClass];

	for (int i = 0; i < epoch; i++) {
		for (int j = 0; j < numTrain; j++) {
			for (int k = 0; k < numClass; k++) {
				prob[k] = 0;
				for (int m1 = 0; m1 < dim + 1; m1++) {
					prob[k] += theta[k][m1] * dataTrain[j][m1];
				}
			}
			CalcProb(prob);
			for (int m2 = 0; m2 < numClass; m2++) {
				for (int n = 0; n < dim + 1; n++) {
					theta[m2][n] += (alpha * (labelTrain[j][m2] - prob[m2]) * dataTrain[j][n]);
				}
			}
		}	
	}
	delete[] prob;
	for (int i = 0; i < numTrain; i++)  {
		delete[] dataTrain[i];
		delete[] labelTrain[i];
	}
	delete[] dataTrain;
	delete[] labelTrain;
	//  
	cout << "INFO    Training done..." << endl;

	
	if (saveModel == 1) {
		ofstream modelFile;
		time_t t = time(NULL); 
		char curTime[24]; 
		strftime(curTime, sizeof(curTime), "%Y%m%d%H%M%S",localtime(&t) ); 
		string _curTime;
		_curTime.assign(curTime);
		char format[5] = ".txt";
		strcat(curTime,format);
		modelFile.open(curTime);

		for (int i = 0; i < numClass; i++) {
			for (int j = 0; j < dim + 1; j++) {
				modelFile<<theta[i][j];
				if(j != dim ) 
					modelFile<<"	";
			}
				modelFile <<"\r\n";
		}
		modelFile << endl;
		modelFile << "Time = "<<_curTime<<endl;
		modelFile << "Training Filename = "<<filenameTrain<<endl;
		modelFile << "Epoch = "<<epoch<<endl;
		modelFile << "Learning Rate = "<<alpha<<endl;
		modelFile.close();
			cout << "INFO    Training parameter saved in " << "\""<<curTime <<"\"..."<<endl ;
	}
	//cout << "------------ ------------ -------------"<<endl<<endl;
}


void SoftmaxReg::CalcProb (float *x) {
	float max = 0.0;
	float sum = 0.0;

	for (int i = 0; i < numClass; i++) if (max < x[i]) max = x[i];
	for (int i = 0; i < numClass; i++) {
		x[i] = exp(x[i] - max);		// avoid overflow
		sum += x[i];
	}
	for (int i = 0; i < numClass; i++) 
		x[i] /= sum;
}


void SoftmaxReg::Predict(const string &filenameTest, bool modelType, const string &filenameModel) {
	float **dataTest, **labelTest;
	int numTest = getNum(filenameTest); 
	dataTest = new float*[numTest];
	labelTest = new float*[numTest];

	for (int i = 0; i < numTest; i++) {
		dataTest[i] = new float[dim + 1];
		labelTest[i] = new float[numClass];
	}
	//load data
	cout<<endl<< "Prediction INFO -------------"<<endl;
	if (loadData(filenameTest, numTest, dataTest, labelTest, false) == 0) {
		cout << "INFO    Error: Predict failed" << endl;
		return;
	}
	
	int count = 0;
	float *predict = new float[numClass];
	float max = 0;
	
	if (modelType == READ_FROM_FILE) {
		fstream fin;
		fin.open(filenameModel, ios::in);
		if (fin.fail()) {
			cerr << "INFO    Error: Open model file failed" <<endl;
			return;
		}
		
		for (int i = 0; i < numClass; i++) {
			for (int j = 0; j < dim+1; j++) {
				fin >> theta[i][j];
			}	
		} 
		fin.close();
		cout << "INFO    Testing data loaded from "<<filenameModel<<"..." <<endl;
	}

	for (int i = 0; i < numTest; i++) {
		for (int j = 0; j < numClass; j++) {
			predict[j] = 0;
			for (int k = 0; k < dim + 1; k++) {
				predict[j] += theta[j][k] * dataTest[i][k];
			}
		}
		CalcProb(predict);
		for (int j = 0; j < numClass; j++) {
			if (predict[j] >= max) max = predict[j];		
		}

		for (int j1 = 0; j1 < numClass; j1++) {
			if (abs(predict[j1] - max) < 0.001 && (int)labelTest[i][j1] == 1) count++;
		}
	}

	delete[] predict;
	for (int i = 0; i < numTest; i++)  {
		delete[] dataTest[i];
		delete[] labelTest[i];
	}
	delete[] dataTest;
	delete[] labelTest;
	
	cout << "INFO    Number of testing samples = "<< numTest <<endl;
	cout << "INFO    Number of samples correctly classified = " << count<<endl;
	cout << "INFO    Classification accuracy = " << (float)count / numTest <<endl;
	//cout << "------------- ------------ -------------"<<endl<<endl;
}
	
