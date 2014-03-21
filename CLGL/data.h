//openGL includes

#include <glload\gl_3_3.h>
#include <glload\gll.hpp>
#include <GL\glfw.h>
#include <glimg\glimg.h>
#include <glimg\ImageCreatorExceptions.h>
#include <glimg\TextureGeneratorExceptions.h>

//#defines
#define PROFILING_OUTPUT_FILE "half_space_results.txt"
#define VERTEX_SHADER_FILE "basicVertex.vert"
#define FRAGMENT_SHADER_FILE "basicFragment.frag"
#define KERNEL_FILE "kernels.cl"
#define TEXTURE_FILE "tex_test.png"


//enums
//Enum for kernels and associated name-strings
typedef enum
{
	RED,
	FILL,
	BOUND_RECT,
	TRIANGLE_SIMPLE,
	TRIANGLE_BOX,
	NUM_KERNELS
}KernelID;

const char *kernelName[] = {	"red",
								"fill",
								"bounding_box",
								"half_space",
								"half_space_box"};
//Enum for CL Buffer Objects
typedef enum
{
	VERTS,
	COLOURS,
	BOUNDS
}BufferID;

//Global constants
//Number of test triangles
static const size_t NUM_TRIANGLES_DEFAULT = 3;

//Window dimensions
static const size_t WIDTH = 800;
static const size_t HEIGHT = 600;

//Associated GL data
GLfloat vertexCoords[] = {	-1.0f, -1.0f, 0.0f,
							-1.0f,  1.0f, 0.0f,
							1.0f,  1.0f, 0.0f,
							1.0f, -1.0f, 0.0f};

GLfloat texCoords[] = {	0.0f, 1.0f,
						0.0f, 0.0f,
						1.0f, 0.0f,
						1.0f, 1.0f};

//Test Triangle Data - replace later with methods to generate data for profiling purposes
int triPixVerts[] = {	300, 200,						
						400, 300,
						500, 200,

						200, 200,						
						100, 500,
						300, 400,

						550, 300,
						500, 500,
						650, 400};

float triColours[] = {	1.0f, 1.0f, 1.0f, 1.0f,
						
						1.0f, 0.5f, 0.0f, 1.0f,
						
						0.1f, 0.9f, 0.8f, 1.0f};
