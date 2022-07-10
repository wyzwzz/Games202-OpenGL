#version 460 core
layout(location = 0) in vec3 iVertexPos;
layout(location = 1) in vec3 iVertexNormal;
layout(location = 2) in vec3 iVertexTangent;
layout(location = 3) in vec2 iTexCoord;

layout(location = 0) out vec3 oVertexPos;
layout(location = 1) out vec3 oVertexNormal;
layout(location = 2) out vec3 oVertexTangent;
layout(location = 3) out vec2 oTexCoord;
layout(location = 4) out float oViewDepth;

uniform mat4 Model;
uniform mat4 ViewModel;
uniform mat4 ProjViewModel;

void main() {
    gl_Position = ProjViewModel * vec4(iVertexPos, 1.0);
    oVertexPos = vec3(Model * vec4(iVertexPos, 1));
    oVertexNormal = normalize(vec3(Model * vec4(iVertexNormal, 0)));
    oVertexTangent = normalize(vec3(Model * vec4(iVertexTangent, 0)));
    oTexCoord = iTexCoord;
    //todo using gl_Position.w
    vec4 view_pos = ViewModel * vec4(iVertexPos, 1.0);
    oViewDepth = view_pos.z / view_pos.w;
}
