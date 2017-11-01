#version 410 core

in vec2 UV;

out vec4 color;

uniform sampler2D renderedTexture;
uniform samplerCube skybox;
uniform mat4 projection;
uniform mat4 view;

const vec3 lightPos = vec3(1.0,1.0,-1.0);
const vec3 ambientColor = vec3(0.1, 0.1, 0.5);
const vec3 diffuseColor = vec3(0.0, 0.0, 0.9);
const vec3 specColor = vec3(1.0, 1.0, 1.0);
const float shininess = 16.0;
const float screenGamma = 2.2; // Assume the monitor is calibrated to the sRGB color space

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

vec3 getNormal(float depth, vec3 eyePos, vec2 invTexScale) {

	vec3 zl = eyePos - uvToEye(UV - vec2(invTexScale.x, 0.0), texture(renderedTexture, UV - vec2(invTexScale.x, 0.0)).x);
	vec3 zr = uvToEye(UV + vec2(invTexScale.x, 0.0), texture(renderedTexture, UV + vec2(invTexScale.x, 0.0)).x) - eyePos;
	vec3 zt = uvToEye(UV + vec2(0.0, invTexScale.y), texture(renderedTexture, UV + vec2(0.0, invTexScale.y)).x) - eyePos;
	vec3 zb = eyePos - uvToEye(UV - vec2(0.0, invTexScale.y), texture(renderedTexture, UV - vec2(0.0, invTexScale.y)).x);

	vec3 dx = zl;
	vec3 dy = zt;

	if (abs(zr.z) < abs(zl.z))
		dx = zr;

	if (abs(zb.z) < abs(zt.z))
		dy = zb;

	vec3 normal = normalize(cross(dx, dy));
	normal = normal*2.f-1.f;

    normal = normalize(normal);

    return normal;
}

vec3 positionFromDepth(float depth, vec2 invTexScale, vec2 texSize) {
    vec2 ndc;             // Reconstructed NDC-space position
    vec3 eye;             // Reconstructed EYE-space position

    float near = 0.1f;
    float far = 100.f;
    float right = texSize.x;
    float top = texSize.y;

    eye.z = near * far / ((depth * (far - near)) - far);

    ndc.x = ((UV.x * invTexScale.x) - 0.5) * 2.0;
    ndc.y = ((UV.y * invTexScale.y) - 0.5) * 2.0;

    eye.x = (-ndc.x * eye.z) * right/near;
    eye.y = (-ndc.y * eye.z) * top/near;

    return eye;
}

void main(){

    float depth = texture(renderedTexture, UV).x;

	if (depth == 0.0f || depth == 1.0f) {
        discard;
	}

    vec2 texSize = textureSize(renderedTexture, 0);
    vec2 invTexScale = 1.f/texSize;
    vec3 eyePos = uvToEye(UV, depth);
    vec3 normal = getNormal(depth, eyePos, invTexScale);

    vec3 lightDir = normalize(lightPos - eyePos);
    float lambertian = max(dot(lightDir,normal), 0.0);
    float specular = 0.0;

    if(lambertian > 0.0) {

        vec3 viewDir = normalize(-eyePos);

    // this is blinn phong
        vec3 halfDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(halfDir, normal), 0.0);
        specular = pow(specAngle, shininess);

    }
    vec3 colorLinear = ambientColor + lambertian * diffuseColor + specular * specColor;

    float reflectance = 0.1f;
    vec3 I = normalize(eyePos);
    vec3 R = reflect(I, normal);
    vec4 cubeMapReflect = texture(skybox, R) * (reflectance + (1.0f - reflectance) * pow(1.0f - dot(I, normal), 5));

    color = vec4(colorLinear, 1.f) * cubeMapReflect;
    // float z = (2 * 0.1) / (100 + 0.1 - depth * (100 - 0.1));
    // color = vec4(z, z, z, 1.f);
  }