#version 410 core

in vec2 UV;
//in vec2 v_blurTexCoords[14];
uniform sampler2D renderedTexture;
uniform mat4 projection;
uniform int direction;

vec3 uvToEye(vec2 p, float z) {
	vec2 pos = p * 2.0f - 1.0f;
	vec4 clipPos = vec4(pos, z, 1.0f);
	vec4 viewPos = inverse(projection) * clipPos;
	return viewPos.xyz / viewPos.w;
}

void main() {
    float depth = texture(renderedTexture, UV).x;
    vec2 screenSize = textureSize(renderedTexture, 0);
    vec3 eyePos = uvToEye(UV, depth);
    float filterRadius = 1.0f/(length(eyePos)) * 40;
    vec2 blurDir;
    float blurScale = 0.1f;
    float blurDepthFalloff = 1.f;

    if (direction == 0) {
        blurDir=vec2(1.0f/screenSize.x,0.0f);
    }
    else {
        blurDir=vec2(0.0f,1.0f/screenSize.y);
    }

    if (depth <= 0.0f) {
   		gl_FragDepth = 0;
   		return;
   	}
   	if (depth >= 1.0f) {
   		gl_FragDepth = depth;
   		return;
   	}

    float sum = 0.0f;
   	float wsum = 0.0f;

   	for (float x = -filterRadius; x <= filterRadius; x += 1.0f) {
   		float s = texture(renderedTexture, UV + x * blurDir).x;

    	if (s >= 1.0f) continue;

    	float r = x * blurScale;
    	float w = exp(-r*r);

    	float r2 = (s - depth) * blurDepthFalloff;
    	float g = exp(-r2*r2);

   		sum += s * w * g;
   		wsum += w * g;
   	}

    if (wsum > 0.0f) {
   		sum /= wsum;
   	}

   	gl_FragDepth = sum;
}