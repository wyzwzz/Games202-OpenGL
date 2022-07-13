#version 460 core
layout(location = 0) in vec3 iVertexPos;

layout(location = 0) out vec3 oVertexPos;


uniform mat4 View;
uniform mat4 Proj;

void main(){
    oVertexPos = iVertexPos;

    //去除view矩阵中平移的部分
    mat4 rotView = mat4(mat3(View));
    vec4 clipPos = Proj * rotView * vec4(iVertexPos,1.f);
    //保证天空盒的片段深度总是为1.0 即最大值
    gl_Position = clipPos.xyww; // xyww / w ==> z = 1.0
}