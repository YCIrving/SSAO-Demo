#version 330 core
in vec2 TexCoord;
uniform sampler2D ourTexture0;
uniform sampler2D ourTexture1;
out vec4 FragColor;
void main()
{
   FragColor = mix(texture(ourTexture0, TexCoord), texture(ourTexture1, TexCoord), 0.2);
}