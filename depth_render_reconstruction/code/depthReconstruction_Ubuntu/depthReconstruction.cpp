#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<vector>
#include<dirent.h>
#include"Eigen/Dense"
#include"Eigen/Core"
#include"opencv2/imgproc.hpp"
#include"opencv2/highgui.hpp"
#include"namepoint.h"
using namespace std;
using namespace cv;

string inputPath;
string outputPath;
vector<Eigen::Vector3d> camList;
int mode = -1;
int resolution = 224;
bool needMesh = false;
float maskThreshold = 0.99;
void usage(string myname)
{
	cout << "Usage: " << myname << " - input[inputpath] - output[outputpath] - resolution[resolution] modeOptions " << endl;
	cout << "Example: " << myname << " -input .\depthImage -output .\results -resolution 224  -reconstruction " << endl;
	cout << "modeOptions:" << endl;
	cout << "-reconstruction	reconstruction from one sample's 20 views " << endl;
	cout << "-interpolation		interpolation between two samples " << endl;
	cout << "-sampling		    randomly generated samples form the model" << endl;
	cout << "other options" << endl;
	cout << "-resolution	set depth image resolution" << endl;
	cout << "-needmesh	mesh output based on Poisson reconstruction, default is deactived" << endl;
	cout << "-mask	threshold for mask, default is 0.99" << endl;
	exit(1);
}

int getdir(string dir, vector<string> &files)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL) {
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL) {
		string filename(dirp->d_name);
		if (filename != "." && filename != "..")
			files.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

vector<vector<Eigen::Vector3d>> evalNormal(vector<vector<Eigen::Vector3d>> samples, vector<Eigen::Vector3d> camPos)
{
	cout << "begin eval normals" << endl;
	vector<vector<Eigen::Vector3d>> normals(samples.size(), vector<Eigen::Vector3d>(0));
	vector<NamedPoint> points;
	int count = 0;
	for (int i = 0; i < samples.size(); i++)
	{
		for (int j = 0; j < samples[i].size(); j++)
		{
			points.push_back(NamedPoint(samples[i][j][0], samples[i][j][1], samples[i][j][2], count));
			count++;
		}
	}
	count = 0;
	PKDTree stree(points.begin(), points.end());
	for (int i = 0; i < samples.size(); i++)
	{
		for (int j = 0; j < samples[i].size(); j++)
		{
			NamedPoint query(samples[i][j][0], samples[i][j][1], samples[i][j][2]);
			BoundedSortedArray<PKDTree::Neighbor> k_closest_elems(20);
			Eigen::Matrix3d local_covariance_matrix = Eigen::Matrix3d::Constant(3, 3, 0);
			Eigen::Vector3d centroid(0, 0, 0);
			stree.kClosestElements<MetricL2>(query, k_closest_elems, -1);  //-1 means there's no limit on the maximum allowed
			for (int kn = 0; kn < k_closest_elems.size(); kn++)
			{
					NamedPoint kpoint = stree.getElements()[k_closest_elems[kn].getIndex()];
					Eigen::Vector3d v(kpoint.position.x, kpoint.position.y, kpoint.position.z);
					centroid += v;
			}
			centroid /= (float)k_closest_elems.size();
			for (int kn = 0; kn < k_closest_elems.size(); kn++)
			{
					NamedPoint kpoint = stree.getElements()[k_closest_elems[kn].getIndex()];
					Eigen::Vector3d v(kpoint.position.x, kpoint.position.y, kpoint.position.z);
					v = v - centroid;
					local_covariance_matrix += v * v.transpose();
			}
			local_covariance_matrix /= (float)k_closest_elems.size();
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>eigensolver(local_covariance_matrix);
			Eigen::Vector3d n = eigensolver.eigenvectors().col(0);
			if (n.dot(camList[i] - samples[i][j]) >= 0){
				normals[i].push_back(n);
			}
			else{
				normals[i].push_back(-n);
			}
			count++;
		}
	}
	return normals;
}

vector<Eigen::Vector3d> denoisePoint(cv::Mat maskimg, Eigen::Vector3d camPos, vector<Eigen::Vector3d> points)
{
	//cout << " denoisePoint  " << endl;
	vector<Eigen::Vector3d> refined;
	Eigen::Affine3d transform(Eigen::Affine3d::Identity());
	Eigen::Vector3d centerTopCam = -camPos;
	Eigen::Vector3d upVector(0, 1, 0);
	centerTopCam.normalize();
	Eigen::Vector3d s = centerTopCam.cross(upVector);
	s.normalize();
	Eigen::Vector3d u = s.cross(centerTopCam);
	u.normalize();
	Eigen::Matrix3d m;
	m << s[0], s[1], s[2], u[0], u[1], u[2], -centerTopCam[0], -centerTopCam[1], -centerTopCam[2];
	Eigen::Vector3d trans;
	trans[0] = s.dot(camPos);
	trans[1] = u.dot(camPos);
	trans[2] = centerTopCam.dot(camPos);
	transform.translate(trans);
	transform.rotate(m);
	float pixel_width_dis = tan(0.5 * 43.0 / 180.0f * M_PI) * (1.0f) / resolution * 2;
	float pixel_hight_dis = tan(0.5*  43.0 / 180.0f * M_PI) * (1.0f) / resolution * 2;
	vector<bool>  pixelMask;
	for (int i = 0; i < maskimg.rows; i++)
	{
		for (int j = 0; j < maskimg.cols; j++)
		{
			Vec3b mintensity = maskimg.at<Vec3b>(i, j);
			float mask = mintensity[0] / 255.0f;
			if (mask > maskThreshold){
				pixelMask.push_back(true);
			}
			else{
				pixelMask.push_back(false);
			}
		}
	}
	for (int i = 0; i < points.size(); i++)
	{
		Eigen::Vector4d pos(points[i][0], points[i][1], points[i][2],1.0);
		Eigen::Vector4d posProj = transform.matrix() * pos;
		posProj[2] = -posProj[2];
		int col_index = posProj[0] / pixel_width_dis / (posProj[2]) + resolution / 2;
		int row_index = resolution / 2 - posProj[1] / pixel_hight_dis / (posProj[2]);
		int index0 = col_index + row_index * resolution;
		int index1 = col_index - 1 + row_index * resolution;
		int index2 = col_index + 1 + row_index * resolution;
		int index3 = col_index + (row_index - 1) * resolution;
		int index4 = col_index + (row_index + 1) * resolution;
		int index5 = col_index -1 + (row_index - 1) * resolution;
		int index6 = col_index - 1 + (row_index + 1) * resolution;
		int index7 = col_index + 1 + (row_index - 1) * resolution;
		int index8 = col_index + 1 +  (row_index + 1) * resolution;
		if ((index0 >= 0 && index0 < pixelMask.size() && pixelMask[index0]) || (index1 >= 0 && index1 < pixelMask.size() && pixelMask[index1]) || (index2 >= 0 && index2 < pixelMask.size() && pixelMask[index2]) ||
			(index3 >= 0 && index3 < pixelMask.size() && pixelMask[index3]) || (index4 >= 0 && index4 < pixelMask.size() && pixelMask[index4]) || (index5 >= 0 && index5 < pixelMask.size() && pixelMask[index5]) ||
			(index6 >= 0 && index6 < pixelMask.size() && pixelMask[index6]) || (index7 >= 0 && index7 < pixelMask.size() && pixelMask[index7]) || (index8 >= 0 && index8 < pixelMask.size() && pixelMask[index8]))
		{
			refined.push_back(points[i]);
		}
	}
	return refined;
}

vector<Eigen::Vector3d> pointfromDepth(cv::Mat img, cv::Mat maskimg, Eigen::Vector3d camPos)
{
	vector<Eigen::Vector3d> sample;
	Eigen::Affine3d transform(Eigen::Affine3d::Identity());
	Eigen::Vector3d centerTopCam = -camPos;
	Eigen::Vector3d upVector(0, 1, 0);
	centerTopCam.normalize();
	Eigen::Vector3d s = centerTopCam.cross(upVector);
	s.normalize();
	Eigen::Vector3d u = s.cross(centerTopCam);
	u.normalize();
	Eigen::Matrix3d m;
	m << s[0], s[1], s[2], u[0], u[1], u[2], -centerTopCam[0], -centerTopCam[1], -centerTopCam[2];
	Eigen::Vector3d trans;
	trans[0] = s.dot(camPos);
	trans[1] = u.dot(camPos);
	trans[2] = centerTopCam.dot(camPos);
	transform.translate(trans);
	transform.rotate(m);
	vector<float> pixelDepth;
	vector<bool>  pixelMask;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			Vec3b intensity = img.at<Vec3b>(i, j);
			float depth = 2.0f - intensity[0] / 255.0f;
			pixelDepth.push_back(depth);
			Vec3b mintensity = maskimg.at<Vec3b>(i, j);
			float mask = mintensity[0] / 255.0f;
			if (mask > 0.995){
				pixelMask.push_back(true);
			}
			else{
				pixelMask.push_back(false);
			}
		}
	}
	Eigen::Vector4d  posTest(0, 0, 0, 1);
	float pixel_width_dis = tan(0.5 * 43.0 / 180.0f * M_PI) * (1.0f) / resolution * 2;
	float pixel_hight_dis = tan(0.5*  43.0 / 180.0f * M_PI) * (1.0f) / resolution * 2;
	for (size_t i = 0; i < pixelDepth.size(); i++)
	{
		if (pixelDepth[i] < 1.75 && pixelMask[i])
		{
			float row_index = resolution - i / resolution;
			float col_index = i % resolution;
			float p_x = (col_index - resolution / 2) * pixel_width_dis * (pixelDepth[i]);
			float p_y = (row_index - resolution / 2) * pixel_hight_dis * (pixelDepth[i]);
			Eigen::Vector4d  pos(p_x, p_y, pixelDepth[i], 1);
			pos[2] = -pos[2];
			pos = transform.inverse().matrix() * pos;
			Eigen::Vector3d posOriginal(pos[0], pos[1], pos[2]);
			sample.push_back(posOriginal);
		}
	}
	return sample;
}

void outputReconstructions(vector<vector<Eigen::Vector3d>> reconstructionSample, int totalSample, string reconstructionModel, vector<vector<Eigen::Vector3d>> normals = vector<vector<Eigen::Vector3d>>(0))
{
	ofstream fout(reconstructionModel);
	fout << "ply" << endl;
	fout << "format ascii 1.0" << endl;
	fout << "element vertex " << totalSample << endl;
	fout << "property float32 x" << endl;
	fout << "property float32 y" << endl;
	fout << "property float32 z" << endl;
	if (normals.size() > 0)
	{
		fout << "property float32 nx" << endl;
		fout << "property float32 ny" << endl;
		fout << "property float32 nz" << endl;
	}
	fout << "element face 0 " << endl;
	fout << "property list uint8 int32 vertex_index" << endl;
	fout << "end_header" << endl;
	for (size_t ii = 0; ii < reconstructionSample.size(); ii++)
	{
		for (size_t jj = 0; jj < reconstructionSample[ii].size(); jj++)
		{
			fout << reconstructionSample[ii][jj][0] << " " << reconstructionSample[ii][jj][1] << " " << reconstructionSample[ii][jj][2] <<" ";
			if (normals.size() > 0)
			{
				fout << normals[ii][jj][0] << " " << normals[ii][jj][1] << " " << normals[ii][jj][2] << " ";
			}
			fout << endl;
		}
	}
}


void reconstructModel(string modelName, string outputName, string nameAppend)
{
	vector<vector<Eigen::Vector3d>> positionAll;
	vector<cv::Mat> maskAll;
	int totalPoint = 0;
	for (size_t i = 0; i < camList.size(); i++)
	{
		string depthName(inputPath);
		depthName.append("/");
		depthName.append(modelName);
		depthName.append(to_string(i));
		depthName.append(nameAppend);
		depthName.append(".png");
		cv::Mat img = cv::imread(depthName);
		string maskName(inputPath);
		maskName.append("//mask//");
		maskName.append(modelName);
		maskName.append(to_string(i));
		maskName.append(nameAppend);
		maskName.append(".png");
		cv::Mat maskimg = cv::imread(maskName);
		maskAll.push_back(maskimg);
		Eigen::Vector3d campos = camList[i];
		positionAll.push_back(pointfromDepth(img, maskimg, campos));
		totalPoint += positionAll.back().size();
	}
	cout << "before denoise " << totalPoint << endl;
	totalPoint = 0;
	for (int i = 0; i < positionAll.size(); i++)
	{
		for (int j = 0; j < maskAll.size(); j++)
		{
			if (j != i)
			{
				//cout << i << " " << j << endl;
				positionAll[i] = denoisePoint(maskAll[j], camList[j], positionAll[i]);
			}
		}
		totalPoint += positionAll[i].size();
	}
	cout << "after denoise " << totalPoint << endl;
	vector<vector<Eigen::Vector3d>> normals(0);
	if (needMesh) normals = evalNormal(positionAll, camList);
	outputReconstructions(positionAll, totalPoint, outputName, normals);
	if (needMesh) 
	{
		cout << "begin Poission Reconstruction" << endl;
		string meshName = outputName.substr(0, outputName.size() - 4) + "_mesh.ply";
		string command = "./PoissonRecon  --in " + outputName + " --out " + meshName;
		system(command.c_str());
	}
}

void reconstruction()
{
	string outModelName;
	vector<int> folderPos;
	for (int i = 0; i < inputPath.size(); i++) {
		if (inputPath[i] == '/') {
			folderPos.push_back(i);
		}
	}
	if (folderPos.back() == inputPath.size() - 1) {
		outModelName = inputPath.substr(folderPos[folderPos.size() - 2] + 1, inputPath.size() - folderPos[folderPos.size() - 2] - 2);
	}
	else {
		outModelName = inputPath.substr(folderPos.back() + 1, inputPath.size() - folderPos.back() - 1);
	}
	string outfile = outputPath + "/" + outModelName;
	vector<string> files = vector<string>();
	getdir(inputPath, files);
	string fileName;
	for (unsigned int i = 0; i < files.size(); i++) {
		vector<int> holdPos;
		for (int j = 0; j < files[i].size(); j++) {
			if (files[i][j] == '-') {
				holdPos.push_back(j);
			}
		}
		fileName = files[i].substr(0, holdPos[holdPos.size() - 2]);
		break;
	}
	reconstructModel(fileName + "-", outfile + "-rec.ply", "-rec");
	reconstructModel(fileName + "-", outfile + "-or.ply", "-or");
}


void interpolationAndSampling()
{
	vector<cv::Mat> imageList;
	vector<cv::Mat> maskList;
	// loadimages
	for (int i = 0; i < camList.size(); i++){
		string fileName = inputPath + "//" + "VP-" + to_string(i) + ".png";
		cv::Mat img = cv::imread(fileName);
		imageList.push_back(img);
	}
	for (int i = 0; i < camList.size(); i++){
		string fileName = inputPath + "//mask//" + "VP-" + to_string(i) + ".png";
		cv::Mat img = cv::imread(fileName);
		maskList.push_back(img);
		//cout << img.rows << " " << img.cols << endl;
	}
	int totalModel = imageList.back().rows * imageList.back().cols / resolution / resolution;
	for (int i = 0; i < totalModel; i++){

		int row = i / (imageList.back().rows / 224);
		int col = i % (imageList.back().rows / 224);
		vector<vector<Eigen::Vector3d>> positionAll;
		int totalPoint = 0;
		string modelName = outputPath + "//" + "model_" + to_string(i) + ".ply";
		for (size_t j = 0; j < camList.size(); j++){
			cv::Rect rect = cv::Rect(row * 224, col * 224, 224, 224);
			cv::Mat imageCut = cv::Mat(imageList[j], rect);
			cv::Mat maskCut = cv::Mat(maskList[j], rect);
			positionAll.push_back(pointfromDepth(imageCut, maskCut, camList[j]));
			totalPoint += positionAll.back().size();
		}
		totalPoint = 0;
		for (int ii = 0; ii < positionAll.size(); ii++)
		{
			for (int j = 0; j < camList.size(); j++)
			{
				if (j != ii)
				{
					cv::Rect rect = cv::Rect(row * 224, col * 224, 224, 224);
					cv::Mat maskCut = cv::Mat(maskList[j], rect);
					positionAll[ii] = denoisePoint(maskCut, camList[j], positionAll[ii]);
				}
			}
			totalPoint += positionAll[ii].size();
		}
		vector<vector<Eigen::Vector3d>> normals(0);
		if (needMesh) normals = evalNormal(positionAll, camList);
		outputReconstructions(positionAll, totalPoint, modelName, normals);
		if (needMesh)
		{
			cout << "begin Poission Reconstruction" << endl;
			string meshName = modelName.substr(0, modelName.size() - 4) + "_mesh.ply";
			string command = "./PoissonRecon  --in " + modelName + " --out " + meshName;
			system(command.c_str());
		}
	}
}

int main(int argc, char **argv)
{
// load configurations
	string camFile("camPosList.txt");
	for (int i = 1; i < argc; i++){
		cout << argv[i] << endl;
		if (strcmp(argv[i], "-input")  == 0){
			inputPath = argv[i + 1];
			outputPath = inputPath;
		}
		if (strcmp(argv[i], "-output") == 0){
			outputPath = argv[i + 1];
		}
		if (strcmp(argv[i],"-resolution") == 0){
			resolution = stoi(argv[i + 1]);
		}
		if (strcmp(argv[i], "-reconstruction") == 0){
			mode = 0;
		}
		if (strcmp(argv[i], "-interpolation") == 0 ){
			mode = 1;
		}
		if (strcmp(argv[i], "-sampling") == 0 ){
			mode = 2;
		}
		if (strcmp(argv[i], "-camPos") == 0 ){
			camFile = argv[i + 1];
		}
		if (strcmp(argv[i], "-needmesh") == 0){
			needMesh = true;
		}
		if (strcmp(argv[i], "-mask") == 0){
			maskThreshold = stof(argv[i + 1]);
		}
	}
	cout << mode << " " << inputPath << " " << outputPath << endl;
	if (mode < 0) usage(argv[0]);
// load camera postions
	double x, y, z;
	ifstream fin(camFile);
	while (fin >> x >> y >> z)
	{
		Eigen::Vector3d p(x*1.5, y*1.5, z*1.5);
		camList.push_back(p);
	}
// do reconstructions
	if (mode == 0)
	{
		reconstruction();
	}
	if (mode == 1 || mode == 2)
	{
		interpolationAndSampling();
	}


}