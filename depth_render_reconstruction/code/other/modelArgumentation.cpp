#include<iostream>
#include<fstream>
#include <string>
#include<sstream>
#include<vector>
#include<ctime>
#include "TriMesh.h"
#include "TriMesh_algo.h"
using namespace std;
using namespace trimesh;

void scaleModel(TriMesh* model, vector<vec3> orginalModel,float scale, int dim, string fileName){
	for (int j = 0; j < model->vertices.size(); j++){
		model->vertices[j][0] = orginalModel[j][dim] * scale;
	}
	model->write(fileName);
}

int main(int argc, char* argv[])
{
	TriMesh *model;
	model = TriMesh::read(argv[1]);
	vector<float> scale = { 1.1f, 1.2f, 1.4f, 0.9f, 0.8f, 0.6f };
	vector<vec3> orginalModel = model->vertices;
	int modelIndex = 0;
	// scale X Y Z
	for (int dim = 0; dim < 3; dim++)
	{
		for (int i = 0; i < scale.size(); i++){
			string fileName(argv[2]);
			fileName = fileName + "_scale_" + to_string(modelIndex) + ".obj";
			scaleModel(model, orginalModel, scale[i], dim, fileName);
			modelIndex++;
		}
	}
	return 0;
}