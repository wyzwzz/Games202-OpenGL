#version 460 core
layout(location = 0) in vec3 iFragColor;

layout(location = 0) out vec4 oFragColor;

void main() {
    oFragColor = vec4(pow(iFragColor,vec3(1.0/2.2)),1.0);
}
