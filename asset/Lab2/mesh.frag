#version 460 core
#extension GL_NV_shader_thread_group : require
layout(location = 0) in vec3 iFragColor;

layout(location = 0) out vec4 oFragColor;


//uniform uint  gl_WarpSizeNV;	// 单个线程束的线程数量
//uniform uint  gl_WarpsPerSMNV;	// 单个SM的线程束数量
//uniform uint  gl_SMCountNV;		// SM数量
//
//in uint  gl_WarpIDNV;		// 当前线程束id
//in uint  gl_SMIDNV;			// 当前线程所在的SM id，取值[0, gl_SMCountNV-1]
//in uint  gl_ThreadInWarpNV;	// 当前线程id，取值[0, gl_WarpSizeNV-1]

void main() {
    oFragColor = vec4(pow(iFragColor,vec3(1.0/2.2)),1.0);
//    oFragColor = vec4(vec3(float(gl_WarpIDNV) / float(gl_WarpsPerSMNV)),1.0);
}
