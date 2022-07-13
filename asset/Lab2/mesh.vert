#version 460 core
layout(location = 0) in vec3 iVertexPos;
layout(location = 1) in vec3 iVertexNormal;
layout(location = 2) in vec2 iTexCoord;

layout(location = 0) out vec3 oColor;

const int MaxSHCount = 25;

layout(std140,binding = 0) uniform LightInfo{
    vec4 LightSH[MaxSHCount];//max degree is 4
};
layout(std430,binding = 0) readonly buffer VertexInfo{
    float VertexSH[];
};

uniform int LightSHCount;// equal to (degree + 1) ^ 2
uniform mat4 Model;
uniform mat4 ProjView;

void main() {
    gl_Position = ProjView * Model * vec4(iVertexPos, 1.0);

    vec3 color = vec3(0);
    int offset = gl_VertexID * MaxSHCount * 3;
    for(int i = 0; i < LightSHCount; ++i)
        color += LightSH[i].rgb * vec3(VertexSH[offset + i * 3],VertexSH[offset + i * 3 + 1],VertexSH[offset + i * 3 + 2]);
    oColor = color;
}
