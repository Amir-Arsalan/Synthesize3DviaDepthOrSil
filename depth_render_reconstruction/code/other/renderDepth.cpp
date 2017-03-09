#include <GL/glew.h>
#include "GL/glut.h"
#include "TriMesh.h"
#include <vector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace trimesh;
using namespace cv;

TriMesh *mesh;
vec3 camPos;
vec3 centerPos;
string imageName;
void display()
{
	int w = glutGet(GLUT_WINDOW_WIDTH);
	int h = glutGet(GLUT_WINDOW_HEIGHT);

	//cout << w << " " << h << endl;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double ar = w / static_cast< double >(h);
	const float zNear = 1;
	const float zFar = 2;
	gluPerspective(43, ar, zNear, zFar); // simulate kinect
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(camPos[0], camPos[1], camPos[2], centerPos[0], centerPos[1], centerPos[2], 0, 1, 0); // set camera position, look at point and up vector
	static float angle = 0;
	glColor3ub(255, 0, 0);
// render mesh 
	for (int it = 0; it < mesh->faces.size(); it++)
	{
		glBegin(GL_TRIANGLES);
		for (int iv = 0; iv < 3; iv++)
		{
			glNormal3f(mesh->normals[mesh->faces[it][iv]][0], mesh->normals[mesh->faces[it][iv]][1], mesh->normals[mesh->faces[it][iv]][2]);
			glVertex3f(mesh->vertices[mesh->faces[it][iv]][0], mesh->vertices[mesh->faces[it][iv]][1], mesh->vertices[mesh->faces[it][iv]][2]);
		}
		glEnd();
	}
	glPopMatrix();
	vector< GLfloat > depth(w * h, 0);
	glReadPixels(0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0]); // read depth buffer
	cv::Mat img(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), CV_32F);// output depth image
	string rawDepth = imageName;
	rawDepth = rawDepth.replace(rawDepth.size() - 3, 3, "txt");
	cout << rawDepth << endl;
	ofstream fout(rawDepth);
	for (int i = 0; i < img.rows; i++ )
	{
		for (int j = 0; j < img.cols; j++)
		{
			depth[i*img.cols + j] = (2.0 * zNear * zFar) / (zFar + zNear - (2.0f * depth[i*img.cols + j] - 1 ) * (zFar - zNear));
			depth[i*img.cols + j] = (depth[i*img.cols + j] - zNear) / (zFar - zNear);
			//cout << depth[i*img.cols + j] << endl;
			//img.at<float>(i, j) = (int)((1.0f - depth[i*img.cols + j] /6) * 255); // flip image
			img.at<float>(i, j) = (1.0f - depth[i*img.cols + j]) * 255;
			//fout << depth[i*img.cols + j] << " ";
		}
		//fout << endl;
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			fout << 1.0f - depth[(img.rows - i - 1) *img.cols + j] << " ";
		}
		fout << endl;
	}
	cv::Mat flipped(img);
	cv::flip(img, flipped, 0);

	cv::Mat imgRGB(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_WINDOW_WIDTH), CV_32FC3);// output depth image
	for (int i = 0; i < imgRGB.rows; i++)
	{
		for (int j = 0; j < imgRGB.cols; j++)
		{
			imgRGB.at<cv::Vec3f>(i, j) = cv::Vec3f(img.at<float>(i, j), img.at<float>(i, j), img.at<float>(i, j)); // flip image
			//cout << imgRGB.at<cv::Vec3f>(i, j) << " " << img.at<float>(i, j) << endl;
		}
	}

	cv::imwrite(imageName, imgRGB);

	glutSwapBuffers();
	exit(0);
	static GLuint tex = 0;
	if (tex > 0)
		glDeleteTextures(1, &tex);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_FLOAT, &depth[0]);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, w, 0, h, -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glEnable(GL_TEXTURE_2D);
	glColor3ub(255, 255, 255);
	glScalef(0.3, 0.3, 1);
	glBegin(GL_QUADS);
	glTexCoord2i(0, 0);
	glVertex2i(0, 0);
	glTexCoord2i(1, 0);
	glVertex2i(w, 0);
	glTexCoord2i(1, 1);
	glVertex2i(w, h);
	glTexCoord2i(0, 1);
	glVertex2i(0, h);
	glEnd();
	glDisable(GL_TEXTURE_2D);
	waitKey(0);
	glutSwapBuffers();
}

void timer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(16, timer, 0);
}

int main(int argc, char **argv)
{
	mesh = TriMesh::read(argv[9]);
	imageName = string(argv[10]);
	camPos[0] = stof(argv[3]); camPos[1] = stof(argv[4]); camPos[2] = stof(argv[5]);
	centerPos[0] = stof(argv[6]); centerPos[1] = stof(argv[7]); centerPos[2] = stof(argv[8]);
	mesh->need_normals();
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(stoi(argv[1]), stoi(argv[2]));
	glutCreateWindow("GLUT");
	glewInit();
	glutDisplayFunc(display);
	glutTimerFunc(0, timer, 0);
	glEnable(GL_DEPTH_TEST);
	glutMainLoop();
	return 0;
}