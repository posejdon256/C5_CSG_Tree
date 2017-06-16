#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <fstream>
#include <cmath>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include "glad\glad.h"
#include <GLFW\glfw3.h>
#include <cuda_gl_interop.h>
#include <surface_functions.h>
#define BUFFER_DEPTH 16
#define MAX_NODES 31
using namespace std;
using namespace thrust;

#pragma region BASE_STRUCTS
struct Vec3
{
	float x, y, z;
	__device__ __host__ Vec3() {}
	__device__ __host__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
	__device__ __host__ Vec3 operator + (Vec3& v)
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	__device__ __host__ Vec3 operator - (Vec3& v)
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	__device__ __host__ Vec3 operator * (float d)
	{
		return Vec3(x * d, y * d, z * d);
	}
	__device__ __host__ Vec3 operator * (Vec3& v)
	{
		return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	__device__ __host__ Vec3 operator / (float d)
	{
		return Vec3(x / d, y / d, z / d);
	}
	__device__ __host__ Vec3 normalize()
	{
		float n = sqrt(x * x + y * y + z * z);
		return Vec3(x / n, y / n, z / n);
	}
	__device__ __host__ float len()
	{
		return sqrt(x * x + y * y + z * z);
	}
};

__device__ __host__ float dot(Vec3 a, Vec3 b)
{
	return (a.x * b.x + a.y * b.y + a.z * b.z);
}

struct Ray
{
	Vec3 o, d;
	__device__ __host__ Ray(Vec3& o, Vec3& d) : o(o), d(d) {}
};

struct SNode
{
	float from;
	float to;
	int32_t sphereInd;
	__device__ __host__ SNode() {}
	__device__ __host__ SNode(float from, float to, int32_t sphereInd) : from(from), to(to), sphereInd(sphereInd) {}
};

struct Sphere
{
	Vec3 c;
	float r;
	Vec3 col;
	__device__ __host__ Sphere() {}
	__device__ __host__ Sphere(Vec3& c, float r, Vec3& col) : c(c), r(r), col(col) {}
	__device__ __host__ Vec3 getNormal(Vec3& pi)
	{
		return (pi - c) / r;
	}
	__device__ __host__ bool intersect(Ray& ray, float &t)
	{
		Vec3 o = ray.o;
		Vec3 d = ray.d;
		Vec3 oc = o - c;
		float b = 2 * dot(oc, d);
		float c = dot(oc, oc) - r*r;
		float disc = b * b - 4 * c;
		if (disc < 1e-4)
			return false;
		disc = sqrt(disc);
		float t0 = -b - disc;
		float t1 = -b + disc;
		t = (t0 < t1) ? t0 : t1;
		return true;
	}
	__device__ __host__ void intersect2(Ray& ray, int32_t &sphereId, SNode *&s)
	{
		Vec3 o = ray.o;
		Vec3 d = ray.d;
		Vec3 oc = o - c;
		float b = 2 * dot(oc, d);
		float c = dot(oc, oc) - r*r;
		float disc = b * b - 4 * c;
		if (disc < 1e-4)
			return;
		disc = sqrt(disc);
		float t0 = -b - disc;
		float t1 = -b + disc;
		float from = (t0 < t1) ? t0 : t1;
		float to = (t0 < t1) ? t1 : t0;
		s->from = from; s->to = to; s->sphereInd = sphereId;
	}
	__device__ __host__ bool sphereIntersect(Sphere& s2)
	{
		Vec3 delta = s2.c - c;
		float dist = sqrt(dot(delta, delta));
		return dist <= r + s2.r;
	}
};

struct Pixel
{
	uint8_t r;
	uint8_t g;
	uint8_t b;
	__device__ __host__ Pixel() : r(0), g(0), b(0) {}
	__device__ __host__  Pixel(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}
};

union RGBX24
{
	uint1 b32;
	struct
	{
		unsigned  r : 8;
		unsigned  g : 8;
		unsigned  b : 8;
		unsigned  na : 8;
	};
};

__device__ __host__ void vecToUINT1(Vec3& pix_col, uint1& pix)
{
	union RGBX24 rgbx;
	rgbx.r = (uint8_t)((pix_col.x > 255) ? 255 : (pix_col.x < 0) ? 0 : pix_col.x);
	rgbx.g = (uint8_t)((pix_col.y > 255) ? 255 : (pix_col.y < 0) ? 0 : pix_col.y);
	rgbx.b = (uint8_t)((pix_col.z > 255) ? 255 : (pix_col.z < 0) ? 0 : pix_col.z);
	rgbx.na = 255;
	pix = rgbx.b32;
}

struct Surfaces
{
	int count;
	int index;
	GLuint* frameBuf;
	GLuint* renderBuf;
	cudaArray_t* cArray;
	cudaGraphicsResource_t* grResource;
	__host__ Surfaces(int count, int height, int width) : count(count), index(0)
	{
		frameBuf = new GLuint[count];
		renderBuf = new GLuint[count];
		grResource = new cudaGraphicsResource_t[count];
		cArray = new cudaArray_t[count];
		glCreateRenderbuffers(count, renderBuf);
		glCreateFramebuffers(count, frameBuf);
		for (int i = 0; i < count; i++)
		{
			glNamedFramebufferRenderbuffer(frameBuf[i], GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderBuf[i]);
			glNamedRenderbufferStorage(renderBuf[i], GL_RGBA8, width, height);
			cudaGraphicsGLRegisterImage(&grResource[i], renderBuf[i], GL_RENDERBUFFER, cudaGraphicsRegisterFlagsSurfaceLoadStore |
				cudaGraphicsRegisterFlagsWriteDiscard);
		}
		cudaGraphicsMapResources(count, grResource, 0);
		for (int i = 0; i < count; i++)
			cudaGraphicsSubResourceGetMappedArray(&cArray[i], grResource[i], 0, 0);
		cudaGraphicsUnmapResources(0, grResource, 0);
	}
	__host__ ~Surfaces()
	{
		for (int i = 0; i < count; i++)
			if (grResource[i] != NULL)
				cudaGraphicsUnregisterResource(grResource[i]);
		glDeleteRenderbuffers(count, renderBuf);
		glDeleteFramebuffers(count, frameBuf);
		delete(frameBuf);
		delete(renderBuf);
		delete(grResource);
		delete(cArray);
	}
};

enum Operation
{
	Empty,
	Leaf,
	Union,
	Intersection,
	Minus
};

struct Node
{
	int32_t index;
	Operation operation;
	int32_t sphereInd;
	__device__ __host__ Node()
	{
		operation = Empty;
		index = 0;
	}
	__device__ __host__ Node(Operation op, int32_t ind)
	{
		operation = op;
		index = ind;
	}
	__device__ __host__ Node(Operation op, int32_t ind, int32_t sInd)
	{
		operation = op;
		index = ind;
		sphereInd = sInd;
	}
	__device__ __host__ int32_t LeftChild() { return index * 2 + 1; }
	__device__ __host__ int32_t RightChild() { return index * 2 + 2; }
	__device__ __host__ int32_t Parent() { return index > 0 ? (index - 1) / 2 : -1; }
};

__device__ __host__ int32_t intPow(int32_t a, int32_t b)
{
	if (b == 0)
		return 1;
	int32_t c = a;
	for (int i = 1; i < b; i++)
		c *= a;
	return c;
}

struct Tree
{
	int32_t count;
	int32_t h;
	__device__ __host__ Tree()
	{
		count = h = 0;
	}
	__device__ __host__ Tree(int32_t h) : h(h)
	{
		count = intPow(2, h + 1) - 1;
	}
};

struct BufferStruct
{
	SNode s[MAX_NODES][BUFFER_DEPTH];
	SNode tmp[BUFFER_DEPTH * 2];
	int32_t sCounts[MAX_NODES];
	int32_t tmpCount;
};
#pragma endregion

__device__ __host__ void ListsMinus(SNode(&s)[MAX_NODES][BUFFER_DEPTH], int32_t *sCounts,
	SNode *tmp, int32_t left, int32_t right, int32_t *tmpCount)
{
	if (sCounts[left] == 0) { *tmpCount = 0; return; }
	if (sCounts[right] == 0)
	{
		*tmpCount = sCounts[left];
		for (int i = 0; i < *tmpCount; i++)
			tmp[i] = s[left][i];
		return;
	}
	int32_t i = 0, j = 0, k = -1;
	float af = s[left][0].from, bf = s[right][0].from;
	while (i != sCounts[left] && j != sCounts[right])
	{
		if (af > s[right][j].to)
		{
			if (j + 1 == sCounts[right])
				break;
			j++;
			bf = s[right][j].from;
			continue;
		}
		if (af > bf)
		{
			if (s[left][i].to < s[right][j].to)
			{
				if (i + 1 == sCounts[left])
					break;
				i++;
				af = s[left][i].from;
				continue;
			}
			af = s[right][j].to;
			if (j + 1 == sCounts[right])
				break;
			j++;
			bf = s[right][j].from;
			continue;
		}
		else
		{
			k++;
			tmp[k].from = af; tmp[k].sphereInd = s[left][i].sphereInd;
			
			if (s[left][i].to < bf)
			{
				tmp[k].to = s[left][i].to;
				if (i + 1 == sCounts[left])
					break;
				i++;
				af = s[left][i].from;
				continue;
			}
			tmp[k].to = bf;
			if (s[left][i].to < s[right][j].to)
			{
				if (i + 1 == sCounts[left])
					break;
				i++;
				af = s[left][i].from;
				continue;
			}
			else
			{
				af = s[right][j].to;
				if (j + 1 == sCounts[right])
					break;
				j++;
				bf = s[right][j].from;
				continue;
			}
		}
	}
	
	while (i != sCounts[left])
	{
		k++;
		tmp[k].from = af; tmp[k].sphereInd = s[left][i].sphereInd;
		tmp[k].to = s[left][i].to;
		i++;
		if (i >= sCounts[left])
			break;
		af = s[left][i].from;
	}
	
	*tmpCount = k < 0 ? 0 : k + 1;
}

__device__ __host__ void ListsIntersection(SNode(&s)[MAX_NODES][BUFFER_DEPTH], int32_t *sCounts,
	SNode *tmp, int32_t left, int32_t right, int32_t *tmpCount)
{
	if (sCounts[left] == 0 || sCounts[right] == 0) { *tmpCount = 0; return; }
	int32_t i = 0, j = 0, k = -1;
	float af = s[left][0].from, bf = s[right][0].from;
	while (i != sCounts[left] && j != sCounts[right])
	{
		if (s[left][i].to < bf)
		{
			if (i + 1 == sCounts[left])
				break;
			i++;
			af = s[left][i].from;
			continue;
		}
		if (s[right][j].to < af)
		{
			if (j + 1 == sCounts[right])
				break;
			j++;
			bf = s[right][j].from;
			continue;
		}

		k++;
		if (af < bf)
		{
			af = bf;
			tmp[k].from = bf; tmp[k].sphereInd = s[right][j].sphereInd;
		}
		else
		{
			bf = af;
			tmp[k].from = af; tmp[k].sphereInd = s[left][i].sphereInd;
		}
		if (s[left][i].to < s[right][j].to)
		{
			bf = tmp[k].to = s[left][i].to;
			if (i + 1 == sCounts[left])
				break;
			i++;
		}
		else
		{
			af = tmp[k].to = s[right][j].to;
			if (j + 1 == sCounts[right])
				break;
			j++;
		}
	}
	*tmpCount = k < 0 ? 0 : k + 1;
}

__device__ __host__ void ListsUnion(SNode(&s)[MAX_NODES][BUFFER_DEPTH], int32_t *sCounts, 
	SNode *tmp, int32_t left, int32_t right, int32_t *tmpCount)
{
	if (sCounts[left] == 0 && sCounts[right] == 0) {*tmpCount = 0; return;}
	if (sCounts[left] == 0) 
	{
		*tmpCount = sCounts[right];
		for (int i = 0; i < *tmpCount; i++)
			tmp[i] = s[right][i];
		return;
	}
	if (sCounts[right] == 0) 
	{
		*tmpCount = sCounts[left];
		for (int i = 0; i < *tmpCount; i++)
			tmp[i] = s[left][i];
		return;
	}
	int32_t n = sCounts[left] + sCounts[right];
	int32_t i = sCounts[left] - 1, j = sCounts[right] - 1, k = n;
	while (k > 0)
		tmp[--k] = (j < 0 || (i >= 0 && s[left][i].from >= s[right][j].from)) ? s[left][i--] : s[right][j--];
	k = 0;
	for (i = 0; i < n; i++)
	{
		if (k != 0 && tmp[k - 1].from <= tmp[i].to)
			while (k != 0 && tmp[k - 1].from <= tmp[i].to)
			{
				tmp[k - 1].to = max(tmp[k - 1].to, tmp[i].to);
				tmp[k - 1].from = min(tmp[k - 1].from, tmp[i].from);
				k--;
			}
		else
			tmp[k] = tmp[i];
		k++;
	}
	*tmpCount = k;
}

__device__ __host__ void visit(Tree *t, Node *nodes, SNode(&s)[MAX_NODES][BUFFER_DEPTH], 
	int32_t *sCounts, SNode *tmp, int32_t *tmpCount, Ray &ray, Sphere *spheres, int32_t index)
{
	if (index >= t->count || nodes[index].operation == Empty || nodes[index].operation == Leaf)
		return;
	int32_t left = nodes[index].LeftChild(), right = nodes[index].RightChild();
	if (nodes[left].operation == Leaf)
	{
		tmp->sphereInd = -1;
		spheres[nodes[left].sphereInd].intersect2(ray, nodes[left].sphereInd, tmp);
		sCounts[left] = tmp->sphereInd < 0 ? 0 : 1;
		if (sCounts[left] > 0)
			for (int i = 0; i < sCounts[left]; i++)
				s[left][i] = tmp[i];
	}
	if (nodes[right].operation == Leaf)
	{
		tmp->sphereInd = -1;
		spheres[nodes[right].sphereInd].intersect2(ray, nodes[right].sphereInd, tmp);
		sCounts[right] = tmp->sphereInd < 0 ? 0 : 1;
		if(sCounts[right] > 0)
			for (int i = 0; i < sCounts[right]; i++)
				s[right][i] = tmp[i];
	}
	switch (nodes[index].operation)
	{
	case Union:
		ListsUnion(s, sCounts, tmp, left, right, tmpCount);
		break;
	case Intersection:
		ListsIntersection(s, sCounts, tmp, left, right, tmpCount);
		break;
	case Minus:
		ListsMinus(s, sCounts, tmp, left, right, tmpCount);
		break;
	}
	sCounts[index] = *tmpCount;
	for (int i = 0; i < sCounts[index]; i++)
		s[index][i] = tmp[i];
}

__device__ __host__ void postOrder(Tree *t, Node *nodes, SNode (&s)[MAX_NODES][BUFFER_DEPTH],
	int32_t *sCounts, SNode *tmp, int32_t *tmpCount, Ray &ray, Sphere *spheres, int32_t index)
{
	if (index >= t->count)
		return;
	postOrder(t, nodes, s, sCounts, tmp, tmpCount, ray, spheres, nodes[index].LeftChild());
	postOrder(t, nodes, s, sCounts, tmp, tmpCount, ray, spheres, nodes[index].RightChild());
	visit(t, nodes, s, sCounts, tmp, tmpCount, ray, spheres, index);
}

__host__ void generateSpheres(host_vector<Sphere> &spheres, int h, int w, int sphereCount)
{
	srand(123456);
	spheres = host_vector<Sphere>(sphereCount);
	for (int i = 0; i < sphereCount; i++)
	{
		Sphere s(
			Vec3(rand() % w, rand() % h, rand() % h / 2),
			h / 20 + (rand() % h / 10),
			Vec3(rand() % 256, rand() % 256, rand() % 256));
		bool isOk = true;
		for (int j = 0; j < i; j++)
			if (spheres[j].sphereIntersect(s))
			{
				isOk = false;
				break;
			}
		if (isOk)
			spheres[i] = s;
		else i--;
	}
}

__host__ bool intersectArray(host_vector<Sphere> &spheres, Ray &ray, float &t, int &id, int spheresCount)
{
	bool intersects = false;
	float d = 0;
	t = 1e10;
	for (int i = 0; i < spheresCount; i++)
	{
		intersects = spheres[i].intersect(ray, d);
		if (intersects && d < t)
		{
			t = d;
			id = i;
		}
	}
	return t < 1e10;
}

__device__ bool intersectArray(Sphere *spheres, Ray ray, float &t, int &id, int spheresCount)
{
	bool intersects = false;
	float d = 0;
	t = 1e10;
	for (int i = 0; i < spheresCount; i++)
	{
		intersects = spheres[i].intersect(ray, d);
		if (intersects && d < t)
		{
			t = d;
			id = i;
		}
	}
	return t < 1e10;
}

surface<void, cudaSurfaceType2D> surf;

__host__ void raytraceHost(cudaArray_t cArray, Tree *t, Node *nodes, Sphere *light, host_vector<Sphere> &_spheres, int h, int w, int spheresCount, cudaStream_t stream)
{
	BufferStruct *buf = new BufferStruct();
	uint1* tab = (uint1*)calloc(h*w, sizeof(uint1));
	Vec3 black(0, 0, 0);
	Sphere *spheres = raw_pointer_cast(_spheres.data());
	for (int y = 1; y < h; ++y)
	{
		for (int x = 0; x < w; ++x)
		{
			Vec3 pix_col(0, 0, 0);
			Ray ray(Vec3(x, y, -300), Vec3(0, 0, 1));
			postOrder(t, nodes, buf->s, buf->sCounts, buf->tmp, &(buf->tmpCount), ray, spheres, 0);
			if (buf->sCounts[0] > 0)
			{
				Vec3 crossP = ray.o + ray.d * buf->s[0][0].from;
				Vec3 lightDir = (*light).c - ray.o;
				Vec3 normal = spheres[buf->s[0][0].sphereInd].getNormal(crossP);
				float lambertian = max(dot(lightDir.normalize(), normal.normalize()), 0.0f);
				pix_col = (spheres[buf->s[0][0].sphereInd].col * lambertian) * 0.7;
				uint1 pix;
				vecToUINT1(pix_col, pix);
				tab[(h - y) * w + x] = pix;
			}
		}
	}
	cudaMemcpyToArray(cArray, 0, 0, tab, h*w * sizeof(uint1), cudaMemcpyHostToDevice);
	delete(tab);
	delete(buf);
}

Tree *p_tree;
Node *p_nodes;
Sphere *p_light;
Sphere *p_spheres;
int32_t delta;
BufferStruct *p_buf;
int32_t tCount;

__global__ void raytraceKernel(Tree *t, Node *nodes, Sphere *spheres, Sphere *light, BufferStruct *p_buf, 
	int32_t tCount, int32_t delta, int h, int w, int spheresCount)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= tCount)
		return;
	BufferStruct *buf = &p_buf[n];
	int x = 0, y = 0;
	for (int i = n * delta; i < (n + 1) * delta; i++)
	{
		x = i / h; 
		y = i % h;
		Ray ray(Vec3(x, y, -300), Vec3(0, 0, 1));
		Vec3 pix_col(0, 0, 0);
		postOrder(t, nodes, buf->s, buf->sCounts, buf->tmp, &(buf->tmpCount), ray, spheres, 0);
		if (buf->sCounts[0] > 0)
		{
			Vec3 crossP = ray.o + ray.d * buf->s[0][0].from;
			Vec3 lightDir = (*light).c - ray.o;
			Vec3 normal = spheres[buf->s[0][0].sphereInd].getNormal(crossP);
			float lambertian = max(dot(lightDir.normalize(), normal.normalize()), 0.0f);
			pix_col = (spheres[buf->s[0][0].sphereInd].col * lambertian) * 0.7;
			uint1 pix;
			vecToUINT1(pix_col, pix);
			surf2Dwrite(pix, surf, x * sizeof(RGBX24), h - y, cudaBoundaryModeZero);
		}
		else
			surf2Dwrite(213000, surf, x * sizeof(RGBX24), h - y, cudaBoundaryModeZero);
	}
}

__host__ void kernelMalloc(cudaArray_const_t cArray, Tree *t, Node *nodes, Sphere *light, 
	host_vector<Sphere> &spheres, int h, int w, int spheresCount, cudaStream_t stream)
{
	//tree
	cudaMalloc((void**)&p_tree, sizeof(Tree));
	cudaMemcpy(p_tree, t, sizeof(Tree), cudaMemcpyHostToDevice);
	//nodes
	cudaMalloc((void**)&p_nodes, t->count * sizeof(Node));
	cudaMemcpy(p_nodes, nodes, t->count * sizeof(Node), cudaMemcpyHostToDevice);
	//light
	cudaMalloc((void**)&p_light, sizeof(Sphere));
	cudaMemcpy(p_light, light, sizeof(Sphere), cudaMemcpyHostToDevice);
	//spheres
	device_vector<Sphere> dev_spheres = spheres;
	p_spheres = raw_pointer_cast(dev_spheres.data());
	//cache arrays
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	tCount = properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor;
	delta = h * w / tCount + ((h*w) % tCount > 0 ? 1 : 0);
	cudaMalloc((void**)&p_buf, tCount * sizeof(BufferStruct));
}

__host__ void raytraceCuda(cudaArray_const_t cArray, Sphere *light, int h, int w, int spheresCount, cudaStream_t stream)
{
	cudaBindSurfaceToArray(surf, cArray);
	cudaMemcpy(p_light, light, sizeof(Sphere), cudaMemcpyHostToDevice);
	cudaMemset(p_buf->sCounts, 0, tCount * sizeof(int32_t));
	dim3 dimBlock = dim3(256, 1, 1);
	dim3 dimGrid = dim3(tCount / 256 + (tCount % 256 > 0 ? 1 : 0), 1, 1);
	
	raytraceKernel << < dimGrid, dimBlock, 0, stream >> > (p_tree, p_nodes, p_spheres, p_light, p_buf, tCount, delta, h, w, spheresCount);
	return;
}

static void pxl_glfw_fps(GLFWwindow* window) //fps counter
{
	// static fps counters
	static double stamp_prev = 0.0;
	static int    frame_count = 0;
	// locals
	const double stamp_curr = glfwGetTime();
	const double elapsed = stamp_curr - stamp_prev;
	if (elapsed > 0.5)
	{
		stamp_prev = stamp_curr;
		const double fps = (double)frame_count / elapsed;
		int  width, height;
		char tmp[64];
		glfwGetFramebufferSize(window, &width, &height);
		sprintf_s(tmp, 64, "(%u x %u) - FPS: %.2f", width, height, fps);
		glfwSetWindowTitle(window, tmp);
		frame_count = 0;
	}
	frame_count++;
}

static void pxl_glfw_error_callback(int error, const char* description)
{
	fputs(description, stderr);
}

static void pxl_glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
}

static void pxl_glfw_init(GLFWwindow** window, const int width, const int height)
{
	//INITIALIZE GLFW/GLAD
	glfwSetErrorCallback(pxl_glfw_error_callback);
	if (!glfwInit())
		exit(EXIT_FAILURE);
	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);
	glfwWindowHint(GLFW_SRGB_CAPABLE, GL_TRUE);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	*window = glfwCreateWindow(width, height, "GLFW / CUDA Interop", NULL, NULL);
	if (*window == NULL)
	{
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(*window);
	//set up GLAD
	gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	//ignore vsync
	glfwSwapInterval(0);
	//only copy RGB
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);
}

int main()
{
#pragma region INIT
	GLFWwindow* window;
	int width = 1280, height = 720, count = 2;
	pxl_glfw_init(&window, width, height);
	cudaSetDevice(0);
	cudaStream_t stream;
	cudaEvent_t  event;
	cudaStreamCreateWithFlags(&stream, cudaStreamDefault);
	cudaEventCreateWithFlags(&event, cudaEventBlockingSync);
	glfwSetKeyCallback(window, pxl_glfw_key_callback);

	clock_t tStart = clock();
	host_vector<Sphere> spheres;
	int h[] = { 720, 900, 1080, 1440 };
	int w[] = { 1280, 1600, 1920, 2560 };
	int sphereCount[] = { 100, 200, 300, 400 };
	Sphere lights[] = {
		Sphere(Vec3(1280, 160, -100), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(500, 600, 150), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(1300, 200, 200), 1, Vec3(255, 255, 255)),
		Sphere(Vec3(2500, 1400, -100), 1, Vec3(255, 255, 255))
	};
	Surfaces* surfaces = new Surfaces(count, height, width);
#pragma endregion

	int choosedI = 0;
	generateSpheres(spheres, h[choosedI], w[choosedI], sphereCount[choosedI]);
	float timelapsed = 0.0f, phi = 0.0f;
	Vec3 diff = (lights[choosedI].c - Vec3(width / 2, height / 2, -100));
	float r = sqrt(diff.x * diff.x + diff.z * diff.z);
	
	Tree *t = new Tree(4);
	Node *nodes = new Node[t->count];
	host_vector<Sphere> sph = host_vector<Sphere>(12);
	sph[0] = Sphere(Vec3(600, 500, 200), 150, Vec3(255, 0, 0));
	sph[1] = Sphere(Vec3(800, 520, 150), 120, Vec3(0, 255, 0));
	sph[2] = Sphere(Vec3(770, 440, 200), 120, Vec3(0, 0, 255));
	sph[3] = Sphere(Vec3(440, 380, 260), 120, Vec3(0, 128, 100));
	sph[4] = Sphere(Vec3(640, 510, -10), 30, Vec3(220, 240, 20));
	sph[5] = Sphere(Vec3(300, 320, 310), 100, Vec3(250, 50, 150));
	sph[6] = Sphere(Vec3(300, 390, 210), 90, Vec3(120, 40, 20));
	sph[7] = Sphere(Vec3(340, 440, 210), 70, Vec3(70, 90, 60));
	sph[8] = Sphere(Vec3(350, 240, 330), 90, Vec3(204, 102, 0));
	sph[9] = Sphere(Vec3(390, 190, 330), 70, Vec3(50, 150, 250));
	sph[10] = Sphere(Vec3(210, 195, 300), 70, Vec3(50, 150, 250));
	sph[11] = Sphere(Vec3(260, 235, 300), 90, Vec3(204, 102, 0));
	

	nodes[0] = Node(Union, 0);
	nodes[1] = Node(Union, 1);
	nodes[2] = Node(Minus, 2);
	nodes[3] = Node(Union, 3);
	nodes[4] = Node(Union, 4);
	nodes[5] = Node(Union, 5);
	nodes[6] = Node(Intersection, 6);
	nodes[7] = Node(Intersection, 7);
	nodes[15] = Node(Leaf, 15, 6);
	nodes[16] = Node(Leaf, 16, 7);

	nodes[8] = Node(Intersection, 8);
	nodes[17] = Node(Leaf, 17, 8);
	nodes[18] = Node(Leaf, 18, 9);

	nodes[9] = Node(Union, 9);
	nodes[19] = Node(Leaf, 19, 5);
	nodes[20] = Node(Leaf, 20, 3);

	nodes[11] = Node(Leaf, 11, 0);
	nodes[12] = Node(Leaf, 12, 4);

	nodes[13] = Node(Leaf, 13, 1);
	nodes[14] = Node(Leaf, 14, 2);

	nodes[10] = Node(Intersection, 10);
	nodes[21] = Node(Leaf, 21, 10);
	nodes[22] = Node(Leaf, 22, 11);

	for(int i = 23; i <= 30; i++)
		nodes[i] = Node(Empty, i);


	kernelMalloc(surfaces->cArray[surfaces->index], t, nodes, &lights[choosedI], sph, height, width, sphereCount[choosedI], stream);

	while (!glfwWindowShouldClose(window))
	{
		tStart = clock() / CLOCKS_PER_SEC;
		pxl_glfw_fps(window); //fps
		cudaGraphicsMapResources(1, &surfaces->grResource[surfaces->index], stream);







		//----------------------------------------------------USE GPU-----------------------------------------------------------
		raytraceCuda(surfaces->cArray[surfaces->index], &lights[choosedI], height, width, sphereCount[choosedI], stream);
		
		
		
		
		
		
		//----------------------------------------------------USE CPU-----------------------------------------------------------
		//raytraceHost(surfaces->cArray[surfaces->index], t, nodes, &lights[choosedI], sph, height, width, sphereCount[choosedI], stream);

	
		
		
		
		
		
		cudaGraphicsUnmapResources(1, &surfaces->grResource[surfaces->index], stream);
		glBlitNamedFramebuffer(surfaces->frameBuf[surfaces->index], 0, 0, 0, width, height, 0, height, width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
		surfaces->index = (surfaces->index + 1) % surfaces->count;
		glfwSwapBuffers(window);
		glfwPollEvents();
		timelapsed += (clock() - tStart) / CLOCKS_PER_SEC;
		if (timelapsed >= 18.0f)
		{
			phi += 0.05f;
			timelapsed = 0.0f;
			float nX = cos(phi) * r + width / 2;
			float nY = lights[choosedI].c.y;
			float nZ = sin(phi) * r - 100;
			lights[choosedI].c = Vec3(nX, nY, nZ);
		}
	}
	//cleanup
	delete(surfaces);
	glfwDestroyWindow(window);
	glfwTerminate();
	cudaDeviceReset();
	exit(EXIT_SUCCESS);
}