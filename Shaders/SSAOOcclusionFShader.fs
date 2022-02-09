#version 330 core
#define MAX_KERNEL_SIZE 128
out float FragColor;
in vec2 TexCoords;

uniform sampler2D gPosition;
uniform sampler2D gNormal;
uniform sampler2D texNoise;

uniform int aoMethod;
uniform int kernelSize;
uniform float radius;
uniform bool rangeCheck;
uniform float ssaoPower;
uniform vec3 samples[MAX_KERNEL_SIZE];
uniform mat4 projection;

float bias = 0.025f;

const vec2 noiseScale = vec2(800.0 / 4.0, 600.0 / 4.0);

void main()
{
   if (aoMethod == 0)
   {
      FragColor = 1;
      return;
   }
   vec3 fragPos = texture(gPosition, TexCoords).xyz;
   vec3 normal = normalize(texture(gNormal, TexCoords).rgb);
   vec3 randomVec = normalize(texture(texNoise, TexCoords * noiseScale).xyz);

   vec3 tangent = normalize(randomVec - normal* dot(randomVec, normal));
   vec3 bitangent = cross(normal, tangent);
   mat3 TBN = mat3(tangent, bitangent, normal);

   float occlusion = 0.0f;

   for (int i=0; i < kernelSize; ++i)
   {
      vec3 samplePos = TBN * samples[i];
      samplePos = fragPos + samplePos * radius;
      vec4 offset = vec4(samplePos, 1.f);
      offset = projection * offset;
      offset.xyz /= offset.w;
      offset.xyz = offset.xyz * 0.5f + 0.5f;
      float sampleDepth = texture(gPosition, offset.xy).z;
      float rangeCheckValue = rangeCheck ? smoothstep(0.f, 1.f, radius / abs(sampleDepth - samplePos.z)) : 1.0f;
      occlusion += ((sampleDepth >= samplePos.z + bias) ? 1.0f : 0.0f) * rangeCheckValue;
   }
   occlusion = 1.0 - (occlusion / kernelSize);
   if (aoMethod == 1 && occlusion >= 0.5f)
   {
      occlusion = 1.0f;
   }
   else
   {
      occlusion = pow(occlusion, ssaoPower);
   }
   FragColor = occlusion;
}