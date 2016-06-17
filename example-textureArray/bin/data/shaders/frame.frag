#version 150

// App uniforms and attributes
uniform sampler2DArray uTexArray;
uniform float uLayer;

in vec2 vTexCoord;

out vec4 vFragColor;

void main()
{
    int prevLayer = int(uLayer);
	int nextLayer = prevLayer + 1;
	float amt = fract(uLayer);
	vFragColor = mix(texture(uTexArray, vec3(vTexCoord, prevLayer)), 
					 texture(uTexArray, vec3(vTexCoord, nextLayer)), 
					 amt);
}
