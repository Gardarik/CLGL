#version 330

uniform sampler2D tex;

in Vertex{
	vec2 texCoord;
} IN;

out vec4 outColour;

void main(void){
	outColour = texture(tex, IN.texCoord);
}