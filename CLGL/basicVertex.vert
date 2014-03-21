#version 330

in vec3 in_pos;
in vec2 in_texCoords;

out Vertex{
	vec2 texCoord;
} OUT;

void main(void){
	gl_Position = vec4(in_pos, 1.0f);
	OUT.texCoord = in_texCoords;
}
