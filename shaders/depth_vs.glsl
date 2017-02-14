#version 410

in vec3 position;

out vec3 posEye;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float pointRadius;
uniform float pointScale;

void main() {
    vec4 viewPos = view * model * vec4(position, 1.0f);
    gl_Position = projection * viewPos;
    posEye = viewPos.xyz;
    gl_PointSize = pointScale*(pointRadius/gl_Position.w);
}
