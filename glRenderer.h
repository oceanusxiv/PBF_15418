//
// Created by Eric Fang on 2/7/17.
//

#ifndef PBF_15418_GLRENDERER_H
#define PBF_15418_GLRENDERER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include <vector>
#include "Camera.h"
#include "ParticleSystem.h"
#include "shader.h"

class glRenderer {
 public:
  glRenderer(int width, int height, Camera &camera, ParticleSystem &sim)
      : depthShader("shaders/depth_vs.glsl",
                    "shaders/depth_fs.glsl"),
        skyBoxShader("shaders/skybox_vs.glsl",
                     "shaders/skybox_fs.glsl"),
        blurShader("shaders/texture_vs.glsl",
                   "shaders/blur_fs.glsl"),
        shadingShader("shaders/texture_vs.glsl",
                      "shaders/shading_fs.glsl"),
        width(width),
        height(height),
        camera(camera),
        simulation(sim) {}
  ~glRenderer();
  int init();
  void onDraw();
#ifdef DEVICE_RENDER
  cudaGraphicsResource_t* resources;
#endif /* DEVICE_RENDER */

 private:
  static inline GLuint loadTexture(GLchar* path);
  static inline GLuint loadCubeMap(std::vector<std::string> faces);
  void setupSkyBox();
  void drawSkyBox(glm::mat4 projection);
  void setupParticles();
  void depthPass(glm::mat4 projection, GLuint FBO, GLuint textureOut);
  void blurPass(glm::mat4 projection, GLuint FBO, GLuint textureIn,
                GLuint textureOut, int direction);
  void shadingPass(glm::mat4 projection, GLuint textureIn);
  void thicknessPass(glm::mat4 projection, GLuint FBO, GLuint textureOut);
  void setupQuad();
  void updateBuffer();
  GLuint setupFBO(GLuint texture);
  GLuint setupTexture();
  Camera& camera;
  ParticleSystem& simulation;
  GLuint particleVAO, particleVBO, skyBoxVAO, skyBoxVBO, cubeMapTexture,
      quadVAO, quadVBO, firstFBO, secondFBO, thicknessTexture, textureOne,
      textureTwo;
  Shader depthShader, skyBoxShader, blurShader, shadingShader;
  int width, height;
  static const GLfloat skyBoxVertices[108];
  static const GLfloat quadVertices[18];
};

#endif  // PBF_15418_GLRENDERER_H
