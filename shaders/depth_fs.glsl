#version 410

in vec3 posEye;

out vec3 color;

uniform mat4 projection;
uniform float pointRadius;

void main() {

    vec3 N;
	N.xy = gl_PointCoord*2.0-1.0;
	float r2 = dot(N.xy,N.xy);
	if (r2 > 1.0) discard;
	N.z = sqrt(1.0 - r2);

	vec4 pixelPos = vec4(posEye + N*pointRadius, 1.0);
	vec4 clipSpacePos = projection * pixelPos;
	float depth = (clipSpacePos.z / clipSpacePos.w)* 0.5f + 0.5f;
    float z = (2 * 0.1) / (100 + 0.1 - depth * (100 - 0.1));
    gl_FragDepth = depth;
	float diffuse = max(0.0, dot(N, vec3(1, -1 , 1)));
	color = vec3(1.0f,0.0f,0.0f) * diffuse;
}
