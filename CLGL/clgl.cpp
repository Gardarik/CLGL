//OpenGL in OpenCL --  software rasterisation on parallel hardware platforms using OpenCL

//Author: Nikolay Sirotinin
//2013

#define _CRT_SECURE_NO_WARNINGS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <exception>
#include <string.h>
#include <vector>
#include <iterator>
#include <ctime>

#include "data.h"

#include <CL\cl.hpp>

//GL Objects
GLuint vertexBufferObj[2];
GLuint vertexArrayObj;
GLuint glProgram;
GLuint glTexObj;
std::auto_ptr<glimg::ImageSet> pImgSet;

//CL Objects
std::vector<cl::Platform> clPlatformList;
std::vector<cl::Device> clDeviceList;
cl::Context clContext;
cl::Program clProgram;
cl::CommandQueue clQueue;
cl::Kernel clKernels[NUM_KERNELS];
cl::Image2D clImg;
std::vector<cl::Buffer> clBufferList;
std::vector<cl::Memory> clInteropList;
std::vector<cl::Event> clWaitList;

using namespace std;

//Pointer to image data
float *imgData;

//Destination pointers for triangle generation
int *vertData;
float *colourData;

//Profiling variables
cl_ulong uStartTime, uEndTime, uTotalTime;

//Mutable global for number of triangles; clumsy but quick
size_t g_numTriangles = NUM_TRIANGLES_DEFAULT;

char* ReadShader(const char* cFileName, size_t* size) {
	//Standard C-like file read for the shaders
	FILE *handle;
	char *cBuffer;

	handle = fopen(cFileName, "r");
	if(handle == NULL)
	{
		printf("%s: failed to open.\n", cFileName);
		exit(EXIT_FAILURE);
	}

	fseek(handle, 0, SEEK_END);
	*size = (size_t)ftell(handle);
	rewind(handle);
	cBuffer = (char*)malloc(*size+1);
	cBuffer[*size] = '\0';
	fread(cBuffer, sizeof(char), *size, handle);
	fclose(handle);

	return cBuffer;
}

std::string ReadKernels(const char* fileName){
	//File reader for kernels, since the CL bindings take strings as arguments
	std::ifstream inFile(fileName);
	std::stringstream fileData;
	fileData << inFile.rdbuf();
	inFile.close();
	return fileData.str();
}

void APIENTRY DebugFunc(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, GLvoid* userParam)
{
	std::string srcName;
	switch(source)
	{
	case GL_DEBUG_SOURCE_API_ARB: srcName = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB: srcName = "Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB: srcName = "Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY_ARB: srcName = "Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION_ARB: srcName = "Application"; break;
	case GL_DEBUG_SOURCE_OTHER_ARB: srcName = "Other"; break;
	}

	std::string errorType;
	switch(type)
	{
	case GL_DEBUG_TYPE_ERROR_ARB: errorType = "Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB: errorType = "Deprecated Functionality"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB: errorType = "Undefined Behavior"; break;
	case GL_DEBUG_TYPE_PORTABILITY_ARB: errorType = "Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE_ARB: errorType = "Performance"; break;
	case GL_DEBUG_TYPE_OTHER_ARB: errorType = "Other"; break;
	}

	std::string typeSeverity;
	switch(severity)
	{
	case GL_DEBUG_SEVERITY_HIGH_ARB: typeSeverity = "High"; break;
	case GL_DEBUG_SEVERITY_MEDIUM_ARB: typeSeverity = "Medium"; break;
	case GL_DEBUG_SEVERITY_LOW_ARB: typeSeverity = "Low"; break;
	}

	printf("%s from %s,\t%s priority\nMessage: %s\n",
		errorType.c_str(), srcName.c_str(), typeSeverity.c_str(), message);
}

GLuint BuildShader (GLenum eShaderType, const char *shaderFileName)
{
	GLuint shader = glCreateShader(eShaderType);
	size_t shaderSize;
	char* shaderSrc = ReadShader(shaderFileName, &shaderSize);
	glShaderSource(shader, 1, (const char**)&shaderSrc, (GLint*)&shaderSize);
	glCompileShader(shader);
	GLint compileStatus;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compileStatus);
	if (compileStatus == GL_FALSE)
	{
		//With ARB_debug_output, we already get the info log on compile failure.
		if(!glext_ARB_debug_output)
		{
			GLint infoLogLength;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

			GLchar *strInfoLog = new GLchar[infoLogLength + 1];
			glGetShaderInfoLog(shader, infoLogLength, NULL, strInfoLog);

			const char *strShaderType = NULL;
			switch(eShaderType)
			{
				case GL_VERTEX_SHADER:
					strShaderType = "vertex"; break;
				case GL_GEOMETRY_SHADER:
					strShaderType = "geometry"; break;
				case GL_FRAGMENT_SHADER:
					strShaderType = "fragment"; break;
			}

			fprintf(stderr, "Compile failure in %s shader:\n%s\n", strShaderType, strInfoLog);
			delete[] strInfoLog;
		}

		throw std::runtime_error("Compile failure in shader.");
	}

	return shader;
}

GLuint BuildProgram(std::vector<GLuint> &shaderList)
{
	GLuint program = glCreateProgram();
	
	for(unsigned short i=0; i < shaderList.size(); i++)
		glAttachShader(program, shaderList[i]);

	//Set default attributes
	glBindAttribLocation(program, 0, "in_pos");
	glBindAttribLocation(program, 1, "in_texCoord");

	glLinkProgram(program);

	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE)
	{
		if(!glext_ARB_debug_output)
		{
			GLint infoLogLength;
			glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

			GLchar *strInfoLog = new GLchar[infoLogLength + 1];
			glGetProgramInfoLog(program, infoLogLength, NULL, strInfoLog);
			fprintf(stderr, "Linker failure: %s\n", strInfoLog);
			delete[] strInfoLog;
		}

		throw std::runtime_error("Failure to link program.");
	}

	//Detach and delete shaders
	for(unsigned short i=0; i<shaderList.size(); i++)
	{
		glDetachShader(program, shaderList[i]);
		glDeleteShader(shaderList[i]);
	}

	return program;
}

void InitGL()
{
	//Initialize GLFW
	if(!glfwInit())
	{
		printf("Failed to initialize GLFW.\n");
		system("pause");
		exit(EXIT_FAILURE);
	}
	
	//Window hints
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef DEBUG
	glfwOpenWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif
	////Set initial window dimensions
	//gWindowWidth = 800;
	//gWindowHeight = 600;

	//Open OpenGL window
	if(!glfwOpenWindow(WIDTH, HEIGHT, 0, 0, 0, 0, 0, 0, GLFW_WINDOW))
	{
		printf("Failed to open window.\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	//Load OpenGL functions
	if(glload::LoadFunctions() == glload::LS_LOAD_FAILED)
	{
		printf("glload fail.\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	//Set title
	glfwSetWindowTitle("CLGL: Triangle Rasterisation");

	if(glext_ARB_debug_output)
	{
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS_ARB);
		glDebugMessageCallbackARB(DebugFunc, (void*)15);
	}
}

void InitCL()
{
	//cl_int err = CL_SUCCESS;
	try
	{
		//Identify platforms
		cl::Platform::get(&clPlatformList);
		//Select first platform with any GPU devices
		for(unsigned int i=0; i<clPlatformList.size(); i++)
		{
			clPlatformList[i].getDevices(CL_DEVICE_TYPE_GPU, &clDeviceList);
			if(!clDeviceList.empty())	break;
		}

		//Set Context Properties: Get associated cl_platform_id using getInfo() on the first GPU
		//Thus conveniently avoiding previous C++ bindings issues :)
		cl_context_properties clProps[] = 
		{
			CL_GL_CONTEXT_KHR,		(cl_context_properties)wglGetCurrentContext(),
			CL_WGL_HDC_KHR,			(cl_context_properties)wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM,	(cl_context_properties)clDeviceList[0].getInfo<CL_DEVICE_PLATFORM>(),
			0
		};
		//Create interop context from GPU devices
		clContext = cl::Context(CL_DEVICE_TYPE_GPU, clProps);

		//Generate program with source and build
		std::string progFile = ReadKernels(KERNEL_FILE);
		cl::Program::Sources clSource(1, std::make_pair(progFile.c_str(), progFile.size()));
		clProgram = cl::Program(clContext, clSource);
		clProgram.build(clDeviceList);
		//Initialize kernels
		for(int i=0; i<NUM_KERNELS; i++)
		{
			clKernels[i] = cl::Kernel(clProgram, kernelName[i]);
		}
		//Create Command Queue with profiling enabled
		clQueue = cl::CommandQueue(clContext, clDeviceList[0], CL_QUEUE_PROFILING_ENABLE);
	}
	catch(cl::Error e)
	{
		cout << "OpenCL initialization failure: " << e.what() << endl
			<< "Error code: " << e.err() << endl;
		if(e.err() == -11)
		{
			std::string clProgLog;
			clProgram.getBuildInfo(clDeviceList[0], CL_PROGRAM_BUILD_LOG, &clProgLog);
			cout << clProgLog;
			system("pause");
			exit(EXIT_FAILURE);
		}
		throw;
	}
}

void InitGLArrays()
{
	//Vertex Array Obect
	glGenVertexArrays(1, &vertexArrayObj);
	glBindVertexArray(vertexArrayObj);

	//VBOs
	glGenBuffers(2, vertexBufferObj);
	//Vertex coordinates
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObj[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	//Texture coordinates
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObj[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	//Unbind
	glBindVertexArray(0);
}

void InitGLShaders()
{
	//Shaders
	std::vector<GLuint> shaders;
	shaders.push_back(BuildShader(GL_VERTEX_SHADER, VERTEX_SHADER_FILE));
	shaders.push_back(BuildShader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER_FILE));

	//Program
	glProgram = BuildProgram(shaders);
	glUseProgram(glProgram);
}

void LoadTextureFromFile()
{
	//Load test texture from file, using glimga
	try{
		cout << "Loading image from texture file..." << endl;
		pImgSet.reset(glimg::loaders::stb::LoadFromFile(TEXTURE_FILE));
		cout << "Success!" << endl;
		//Print image dimensions
		glimg::Dimensions imgDims = pImgSet.get()->GetDimensions();
		cout << "Image dimensions: " << endl
			<< "Width: " << imgDims.width << endl
			<< "Height: " << imgDims.height << endl;
		//Print image format information
		glimg::ImageFormat imgFormat = pImgSet.get()->GetFormat();
		cout << "Image Format:" << endl
			<< "Type: " << imgFormat.Type() << endl
			<< "Components: " << imgFormat.Components() << endl
			<< "Component order: " << imgFormat.Order() <<endl
			<< "Bit Depth: " << imgFormat.Depth() << endl;
	}
	catch(glimg::ImageCreationException e){
		cout << "Image creation failure: " << e.what() << endl;
		throw;
	}
}

void InitGLTexture()
{
	//Allocate host memory for image data
	imgData = new float[4 * WIDTH * HEIGHT];

	//Enable and configure texture
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &glTexObj);
	
	//Provide image and set parameters
	glBindTexture(GL_TEXTURE_2D, glTexObj);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, imgData);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glActiveTexture(GL_TEXTURE0);

	//Unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);


}

void SetCLRenderTarget()
{
	//Bind GL texture object
	glBindTexture(GL_TEXTURE_2D, glTexObj);	
	
	//Create and configure CL interop object
	try
	{
		cl::ImageGL clTexObj(clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glTexObj);
		//Add to interop object list
		clInteropList.push_back(clTexObj);
	}
	catch(cl::Error e)
	{
		cout << "Texture interop failure: " << e.what() << endl
			<< "Error code: " << e.err() << endl;
	}
	//Unbind texture
	glBindTexture(GL_TEXTURE_2D, 0);
}

void GenerateTriangles(unsigned int numTriangles, int hfwd, int ht)
{
	//local variables 
	int vPos, cPos, moveX, moveY;
	//Allocate memory for output
	size_t memSize = numTriangles*3*2;
	vertData = new int[memSize];
	memSize = numTriangles*4;
	colourData = new float[memSize];
	//put base triangle at origin (top-left) first
	vertData[0] = 0;
	vertData[1] = 0;
	vertData[2] = hfwd;
	vertData[3] = ht;
	vertData[4] = hfwd*2;
	vertData[5] = 0;
	//randomize colour (somewhat)
	colourData[0] = (float)((rand()% 10)/10.0f);
	colourData[1] = (float)((rand()% 10)/10.0f);
	colourData[2] = (float)((rand()% 10)/10.0f);
	colourData[3] = 1.0f;
	//Here we go with the big ol' for loop
	for(unsigned int i=1; i<numTriangles; i++){
		//find correct position in dest arrays
		vPos = i*6;
		cPos = i*4;
		//random translation in NDC for vertices
		moveX = int(rand() % (WIDTH - hfwd*2));
		moveY = int(rand() % (HEIGHT - ht));
		//Looks like we're doing it the long way for now, maybe tidy up later...
		vertData[0 + vPos] = 0 + moveX;
		vertData[1 + vPos] = 0 + moveY;

		vertData[2 + vPos] = hfwd+ moveX;
		vertData[3 + vPos] = ht+ moveY;

		vertData[4 + vPos] = hfwd*2+ moveX;
		vertData[5 + vPos] = 0 + moveY;

		//Colour
		for(int i=0; i<3; i++) colourData[cPos +i] = (float)((rand()% 10)/10.0f);
		//set alpha component to 1
		colourData[cPos + 3] = 1.0f;
	}
}

void InitCLBuffers()
{
	char cRep;
	int numTri, hw, ht;
	cout << "Generate some triangle data (y/n)?" << endl;
	cin >> cRep;
	if(cRep == 'y'|| cRep == 'Y'){
		cout << "Enter number of triangles." << endl;
		cin >> numTri;
		g_numTriangles = numTri;
		cout << "Enter triangle half-width value." << endl;
		cin >> hw;
		cout << "Enter triangle height parameter." << endl;
		cin >> ht;
		cout << "Generating..." << endl;
		GenerateTriangles(g_numTriangles, hw, ht);
		cout << "Done." << endl;
		cout << "Creating Buffers..." << endl;
		try
		{
			//Create buffers from triangle data on the host and add to buffer list
			cl::Buffer clVertBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int)*6*g_numTriangles, vertData);
			clBufferList.push_back(clVertBuffer);
			cl::Buffer clColourBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*4*g_numTriangles, colourData);
			clBufferList.push_back(clColourBuffer);
			cl::Buffer clBoundsBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(int)*g_numTriangles*4, NULL);
			clBufferList.push_back(clBoundsBuffer);
		}
		catch(cl::Error e)
		{
			cout << "OpenCL memory object failure: " << e.what() << endl
				<< "Error code: " << e.err() << endl;
		throw;
		}
		cout << "Done!" << endl;
	}
	else{
		cout << "Using hard-coded test data." << endl;
		try
		{
			//Create buffers from triangle data on the host and add to buffer list
			cl::Buffer clVertBuffer(clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(triPixVerts), triPixVerts);
			clBufferList.push_back(clVertBuffer);
			cl::Buffer clColourBuffer(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(triColours), triColours);
			clBufferList.push_back(clColourBuffer);
			cl::Buffer clBoundsBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(int)*g_numTriangles*4, NULL);
			clBufferList.push_back(clBoundsBuffer);
		}
		catch(cl::Error e)
		{
			cout << "OpenCL memory object failure: " << e.what() << endl
				<< "Error code: " << e.err() << endl;
		throw;
		}
	}
}

void SetCLArgs()
{
	//Red kernel
	clKernels[RED].setArg<cl::Memory>(0, clInteropList[0]);
	//Bounding rectangle kernel
	clKernels[BOUND_RECT].setArg<cl::Buffer>(0, clBufferList[VERTS]);
	clKernels[BOUND_RECT].setArg<cl::Buffer>(1, clBufferList[BOUNDS]);
	//Simple triangle kernel
	clKernels[TRIANGLE_SIMPLE].setArg<cl::Buffer>(0, clBufferList[VERTS]);
	clKernels[TRIANGLE_SIMPLE].setArg<cl::Buffer>(1, clBufferList[COLOURS]);
	clKernels[TRIANGLE_SIMPLE].setArg<cl::Memory>(2, clInteropList[0]);
	//Half-space with bounding box
	clKernels[TRIANGLE_BOX].setArg<cl::Buffer>(0, clBufferList[VERTS]);
	clKernels[TRIANGLE_BOX].setArg<cl::Buffer>(1, clBufferList[COLOURS]);
	clKernels[TRIANGLE_BOX].setArg<cl::Memory>(2, clInteropList[0]);
}

void ConfigureData()
{
	cout << "Configuring data..." << endl;
	//Initialize OpenGL objects
	InitGLArrays();
	InitGLTexture();
	InitGLShaders();
	//Configure OpenCL render/interop target
	SetCLRenderTarget();
	//Create CL buffer objects
	cout << "CL Buffers..." << endl;
	InitCLBuffers();
	//Set kernel Arguments
	SetCLArgs();
}

unsigned long int ExecuteKernels()
{
	try
	{
		//Profiling event
		cl::Event profEvent;
		//Make sure OpenGL processing is finished
		glFinish();
		//Get exclusive access to GL texture object
		clQueue.enqueueAcquireGLObjects(&clInteropList);
		//Execute kernels
		clQueue.enqueueNDRangeKernel(clKernels[TRIANGLE_BOX], cl::NullRange, cl::NDRange(WIDTH, HEIGHT, g_numTriangles), cl::NullRange, NULL, &profEvent);
		//Release texture object
		clQueue.enqueueReleaseGLObjects(&clInteropList);
		//Finish OpenCL processing
		clQueue.finish();
		//Get the profiling info
		profEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &uStartTime);
		profEvent.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &uEndTime);
	}
	catch(cl::Error e)
	{
		cout << "Kernel Execution failure: " << e.what() << endl
			<< "Error code: " << e.err() << endl;
		throw;
	}
	unsigned long int exTime = uEndTime - uStartTime;
	return exTime;
}  
//Display function
void Display()
{
	glUniform1i(glGetUniformLocation(glProgram, "tex"), 0);

	glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glBindVertexArray(vertexArrayObj);
	glBindTexture(GL_TEXTURE_2D, glTexObj);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);

	glfwSwapBuffers();
}
//
////Reshape callback function
//void reshape (int w, int h)
//{
//	gWindowWidth = w;
//	gWindowHeight = h;
//
//	glViewport(0, 0, (GLsizei) gWindowWidth, (GLsizei) gWindowHeight);
//}

void Profile(unsigned int iNumFrames, const char *cOutputFile, std::string strMessage)
{
	//Filestream
	fstream output(cOutputFile, std::ios_base::app);
	//Timestamp
	time_t timeStamp;
	//Timer
	clock_t timer;
	//Keeping track of total time taken
	size_t totalTicks = 0;
	unsigned long int totalKernelTime = 0;
	//Check it's there
	if(output.is_open()){
		//Must set output pointer to end of file
		output.seekp(ios_base::end);
		//Timestamp
		time(&timeStamp);
		//Mark profiling session
		output << "Profiling session: " << ctime(&timeStamp) << endl;
		output << "Running " << iNumFrames << " frames." << endl;
		output << "Message: " << strMessage << endl << endl;
		//Begin profiling
		for(unsigned int i = 0; i < iNumFrames; i++){
			//output << "Frame " << i << ": ";
			timer = clock();
			unsigned long int kernelTime;
			kernelTime = ExecuteKernels();
			Display();
			timer = clock() - timer;
			//output << timer << "	ticks.	";
			//output << "Kernel Execution Time: " << kernelTime << endl;
			totalTicks += timer;
			totalKernelTime += kernelTime;
		}
		//Output summary
		double totalSecs = ((double)totalTicks)/CLOCKS_PER_SEC;
		output << endl << "Total time taken: " << totalTicks << " ticks = " << totalSecs << " sec" << endl;
		output << "Average loop time: " << (totalSecs/iNumFrames)*1000000 << " microsec" << endl;
		output << "Average kernel execution time: " << (totalKernelTime/iNumFrames)/1000 << " microsec" <<endl;
		output << "Average frame rate: " << ((double)iNumFrames)/totalSecs << " fps" << endl << endl;
	}
	else{
		cout << "Failed to open the output file.\n";
	}
}

int main(void)
{
	int running = GL_TRUE;
	unsigned int iFrames;
	string strMsg;
	char cResponse;

	//Initialize OpenGL
	cout << "Initializing OpenGL..." << endl;
	InitGL();
	cout << "Complete" << endl;
	//Initialize OpenCL
	cout << "Initializing OpenCL..." << endl;
	InitCL();
	cout << "Complete" << endl;
	//Generate VAO, VBO, load shaders and configure CL-GL interop
	cout << "Configuring interoperability objects..." << endl;
	ConfigureData();
	cout << "Complete" << endl << endl;
	//Set reshape callback
//	glfwSetWindowSizeCallback(reshape);
	//Ask for profiling
	cout << "Enable profiling (Y/N) ?" << endl;
	cin >> cResponse;
	cout << endl;
	if(cResponse == 'y' || cResponse == 'Y'){
		cout << "Enter the number of frames for this profiling session." << endl;
		cin >> iFrames;
		cout << endl << "Enter the message for this profiling session." << endl;
		//Remember do NOT leave blank space in the string when using cin, otherwise it will cut off right there
		getline(cin, strMsg, '.');
		cout << endl << "Profiling..." << endl;
		Profile(iFrames, PROFILING_OUTPUT_FILE, strMsg);
		cout << "Complete" << endl;
	}
	else{
		//Main loop
		while(running){
			ExecuteKernels();
			Display();
			//Check if Esc pressed or window closed
			running = !glfwGetKey(GLFW_KEY_ESC) && glfwGetWindowParam(GLFW_OPENED);
		}
	}
	//Close window and terminate GLFW
	glfwTerminate();
	//Release memory
	delete imgData;
	delete vertData;
	delete colourData;
	//Exit main
	system("pause");
	exit(EXIT_SUCCESS);
}