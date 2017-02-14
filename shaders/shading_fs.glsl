#version 410 core

in vec2 UV;

out vec3 color;

uniform sampler2D renderedTexture;
uniform mat4 projection;

const vec3 lightPos = vec3(1.0,1.0,1.0);
const vec3 ambientColor = vec3(0.0, 0.0, 0.3);
const vec3 diffuseColor = vec3(0.0, 0.0, 0.9);
const vec3 specColor = vec3(1.0, 1.0, 1.0);
const float shininess = 16.0;
const float screenGamma = 2.2; // Assume the monitor is calibrated to the sRGB color space

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main(){

    float depth = texture(renderedTexture,UV).x;

	if (depth == 0.0f) {
        discard;
	}

	if (depth == 1.0) {
		discard;
	}
    vec2 texSize = textureSize(renderedTexture, 0);
    vec2 invTexScale = 1.f/texSize;
    vec3 eyePos = uvToEye(UV, depth);
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
	/*
	float diffuse = max(0.0, dot(normal*2.f-1.f, vec3(-1,1,1)));
	float z = (2 * 0.1) / (100 + 0.1 - texture(renderedTexture,UV).x * (100 - 0.1));

	color=diffuse;*/

    normal = normalize(normal);
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
    vec3 colorLinear = ambientColor +
                     lambertian * diffuseColor +
                     specular * specColor;
  // apply gamma correction (assume ambientColor, diffuseColor and specColor
  // have been linearized, i.e. have no gamma correction in them)
    vec3 colorGammaCorrected = pow(colorLinear, vec3(1.0/screenGamma));
  // use the gamma corrected color in the fragment
    color = colorGammaCorrected;
  }