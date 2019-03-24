//
// Created by Eric Fang on 2/7/17.
//

#include "glRenderer.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define cudaCheck(x)                                                          \
  {                                                                           \
    cudaError_t err = x;                                                      \
    if (err != cudaSuccess) {                                                 \
      printf("Cuda error: %d in %s at %s:%d\n", err, #x, __FILE__, __LINE__); \
      assert(0);                                                              \
    }                                                                         \
  }

const GLfloat glRenderer::skyBoxVertices[108] = {
    // Positions
    -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,

    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,

    1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f,

    -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f,

    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,

    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f,
    1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};

const GLfloat glRenderer::quadVertices[18] = {
    -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f,
    -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
};

GLuint glRenderer::loadTexture(GLchar *path) {
  // Generate texture ID and load texture data
  GLuint textureID;
  glGenTextures(1, &textureID);
  int width, height, n;
  unsigned char *image = stbi_load(path, &width, &height, &n, 0);
  // Assign texture to ID
  glBindTexture(GL_TEXTURE_2D, textureID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, image);
  glGenerateMipmap(GL_TEXTURE_2D);

  // Parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  delete image;
  return textureID;
}

GLuint glRenderer::loadCubeMap(std::vector<std::string> faces) {
  GLuint textureID;
  glGenTextures(1, &textureID);

  int width, height, n;
  unsigned char *image;

  glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
  for (GLuint i = 0; i < faces.size(); i++) {
    image = stbi_load(faces[i].c_str(), &width, &height, &n, 0);
    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, image);
    delete image;
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  return textureID;
}

int glRenderer::init() {
  setupParticles();

  setupSkyBox();

  setupQuad();

  textureOne = setupTexture();

  textureTwo = setupTexture();

  firstFBO = setupFBO(textureOne);

  secondFBO = setupFBO(textureTwo);

  return 0;
}

void glRenderer::onDraw() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glClearColor(0.0f, 1.0f, 1.0f, 1.0f);

  glm::mat4 projection = glm::perspective(
      camera.Zoom, static_cast<float>(width) / static_cast<float>(height), 0.1f,
      1000.f);

  updateBuffer();

  depthPass(projection, firstFBO, textureOne);
  blurPass(projection, secondFBO, textureOne, textureTwo, 0);
  blurPass(projection, firstFBO, textureTwo, textureOne, 1);
  shadingPass(projection, textureOne);

  drawSkyBox(projection);
}

void glRenderer::setupSkyBox() {
  glGenVertexArrays(1, &skyBoxVAO);
  glGenBuffers(1, &skyBoxVBO);
  glBindVertexArray(skyBoxVAO);
  glBindBuffer(GL_ARRAY_BUFFER, skyBoxVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(skyBoxVertices), &skyBoxVertices,
               GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  std::vector<std::string> faces =
      {"skybox/right.jpg", "skybox/left.jpg", "skybox/top.jpg", "skybox/bottom.jpg", "skybox/back.jpg",
       "skybox/front.jpg"};

  cubeMapTexture = loadCubeMap(faces);
}

void glRenderer::drawSkyBox(glm::mat4 projection) {
  // Draw skybox as last
  glDepthFunc(GL_LEQUAL);  // Change depth function so depth test passes when
  // values are equal to depth buffer's content
  skyBoxShader.Use();
  glm::mat4 view =
      glm::mat4(glm::mat3(camera.GetViewMatrix()));  // Remove any translation
  // component of the view
  // matrix
  glUniformMatrix4fv(glGetUniformLocation(skyBoxShader.Program, "view"), 1,
                     GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(skyBoxShader.Program, "projection"),
                     1, GL_FALSE, glm::value_ptr(projection));
  // skybox cube
  glBindVertexArray(skyBoxVAO);
  glActiveTexture(GL_TEXTURE0);
  // glUniform1i(glGetUniformLocation(shader.Program, "skybox"), 0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTexture);
  glDrawArrays(GL_TRIANGLES, 0, 36);
  glBindVertexArray(0);
  glDepthFunc(GL_LESS);  // Set depth function back to default
}

void glRenderer::blurPass(glm::mat4 projection, GLuint FBO, GLuint textureIn,
                          GLuint textureOut, int direction) {
  glBindFramebuffer(GL_FRAMEBUFFER, FBO);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, textureOut, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  blurShader.Use();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureIn);

  glUniform1i(glGetUniformLocation(blurShader.Program, "renderedTexture"), 0);
  glUniformMatrix4fv(glGetUniformLocation(blurShader.Program, "projection"), 1,
                     GL_FALSE, glm::value_ptr(projection));
  glUniform1i(glGetUniformLocation(blurShader.Program, "direction"), direction);

  glBindVertexArray(quadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
}

void glRenderer::setupQuad() {
  glGenBuffers(1, &quadVBO);
  glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices),
               static_cast<const GLvoid *>(quadVertices), GL_STATIC_DRAW);

  glGenVertexArrays(1, &quadVAO);
  glBindVertexArray(quadVAO);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

GLuint glRenderer::setupFBO(GLuint texture) {
  GLuint FBO;

  glGenFramebuffers(1, &FBO);
  glBindFramebuffer(GL_FRAMEBUFFER, FBO);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, texture, 0);
  // No color output in the bound framebuffer, only depth.
  glDrawBuffer(GL_NONE);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "framebuffer not ok!" << std::endl;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  return FBO;
}

GLuint glRenderer::setupTexture() {
  GLint dims[4] = {0};
  glGetIntegerv(GL_VIEWPORT, static_cast<GLint *>(dims));
  GLint fbWidth = dims[2];
  GLint fbHeight = dims[3];

  GLuint texture;

  glGenTextures(1, &texture);

  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, fbWidth, fbHeight, 0,
               GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE,
  // GL_COMPARE_R_TO_TEXTURE);
  glBindTexture(GL_TEXTURE_2D, 0);

  return texture;
}

void glRenderer::setupParticles() {
  glGenBuffers(1, &particleVBO);
  glBindBuffer(GL_ARRAY_BUFFER, particleVBO);

#ifdef DEVICE_RENDER
  glBufferData(GL_ARRAY_BUFFER, simulation.getParticleNum() * sizeof(float) * 3,
               0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGraphicsGLRegisterBuffer(resources, particleVBO,
                               cudaGraphicsRegisterFlagsNone);
#else
  glBufferData(GL_ARRAY_BUFFER, simulation.getParticleNum() * sizeof(float) * 3,
               simulation.getParticlePos(), GL_STREAM_DRAW);

  glGenVertexArrays(1, &particleVAO);
  glBindVertexArray(particleVAO);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
#endif /* DEVICE_RENDER */
}

void glRenderer::updateBuffer() {
// Update the VBO
#ifdef DEVICE_RENDER
  void* particlePosPtr;
  cudaCheck(cudaGraphicsMapResources(1, resources));
  size_t size;

  cudaGraphicsResourceGetMappedPointer(&particlePosPtr, &size, resources[0]);
  float* pos = simulation.getParticlePos();
  cudaCheck(cudaMemcpy(particlePosPtr, pos,
                       simulation.getParticleNum() * sizeof(float) * 3,
                       cudaMemcpyDeviceToDevice));

  cudaGraphicsUnmapResources(1, resources);

#else

  glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
  glBufferSubData(GL_ARRAY_BUFFER, 0,
                  simulation.getParticleNum() * sizeof(float) * 3,
                  simulation.getParticlePos());

#endif /* DEVICE_RENDER */
}

void glRenderer::depthPass(glm::mat4 projection, GLuint FBO,
                           GLuint textureOut) {
  depthShader.Use();

  // Create camera transformation
  glm::mat4 model;
  glm::mat4 view = camera.GetViewMatrix();

  GLfloat pointRadius = 2.0f;
  GLfloat pointScale = 1000.0f;

  // Pass the matrices to the shader
  glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "model"), 1,
                     GL_FALSE, glm::value_ptr(model));
  glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "view"), 1,
                     GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "projection"), 1,
                     GL_FALSE, glm::value_ptr(projection));
  glUniform1f(glGetUniformLocation(depthShader.Program, "pointRadius"),
              pointRadius);
  glUniform1f(glGetUniformLocation(depthShader.Program, "pointScale"),
              pointScale);

  glBindFramebuffer(GL_FRAMEBUFFER, FBO);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, textureOut, 0);

  glBindVertexArray(particleVAO);
  glDrawArrays(GL_POINTS, 0, simulation.getParticleNum());
  glBindVertexArray(0);

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void glRenderer::shadingPass(glm::mat4 projection, GLuint textureIn) {
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  shadingShader.Use();

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, textureIn);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTexture);

  // glm::mat4 view = camera.GetViewMatrix();

  glUniform1i(glGetUniformLocation(shadingShader.Program, "renderedTexture"),
              0);
  glUniform1i(glGetUniformLocation(shadingShader.Program, "skybox"), 1);
  // glUniformMatrix4fv(glGetUniformLocation(depthShader.Program, "view"), 1,
  // GL_FALSE, glm::value_ptr(view));
  glUniformMatrix4fv(glGetUniformLocation(shadingShader.Program, "projection"),
                     1, GL_FALSE, glm::value_ptr(projection));

  glBindVertexArray(quadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
}

glRenderer::~glRenderer() {
#ifdef DEVICE_RENDER
  cudaDeviceSynchronize();
  cudaGraphicsUnregisterResource(*resources);
#endif /* DEVICE_RENDER */

  glDeleteVertexArrays(1, &particleVAO);
  glDeleteVertexArrays(1, &skyBoxVAO);
  glDeleteBuffers(1, &particleVBO);
  glDeleteBuffers(1, &skyBoxVBO);
}
