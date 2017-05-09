//
// Created by Eric Fang on 2/7/17.
//

#ifndef PBF_15418_GLRENDERER_H
#define PBF_15418_GLRENDERER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>
#include "SOIL2.h"
#include <vector>
#include "shader.h"
#include "Camera.h"
#include "particleSystem.h"


class glRenderer {

public:
    glRenderer(int width, int height, Camera& camera, particleSystem &sim, std::string srcPath) :
            depthShader(srcPath + "shaders/depth_vs.glsl", srcPath + "shaders/depth_fs.glsl"),
            skyBoxShader(srcPath + "shaders/skybox_vs.glsl", srcPath + "shaders/skybox_fs.glsl"),
            blurShader(srcPath + "shaders/texture_vs.glsl", srcPath + "shaders/blur_fs.glsl"),
            shadingShader(srcPath + "shaders/texture_vs.glsl", srcPath + "shaders/shading_fs.glsl"),
            width(width),
            height(height),
            camera(camera),
            simulation(sim),
            srcPath(srcPath) {}
    ~glRenderer();
    int init();
    void onDraw();

private:
    static inline GLuint loadTexture(GLchar* path)
    {
        //Generate texture ID and load texture data
        GLuint textureID;
        glGenTextures(1, &textureID);
        int width,height;
        unsigned char* image = SOIL_load_image(path, &width, &height, 0, SOIL_LOAD_RGB);
        // Assign texture to ID
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
        glGenerateMipmap(GL_TEXTURE_2D);

        // Parameters
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glBindTexture(GL_TEXTURE_2D, 0);
        SOIL_free_image_data(image);
        return textureID;
    }
    static inline GLuint loadCubeMap(std::vector<std::string> faces)
    {
        GLuint textureID;
        glGenTextures(1, &textureID);

        int width,height;
        unsigned char* image;

        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
        for(GLuint i = 0; i < faces.size(); i++)
        {
            image = SOIL_load_image(faces[i].c_str(), &width, &height, 0, SOIL_LOAD_RGB);
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
            SOIL_free_image_data(image);
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

        return textureID;
    }
    void setupSkyBox();
    void drawSkyBox(glm::mat4 projection);
    void setupParticles();
    void depthPass(glm::mat4 projection, GLuint FBO, GLuint textureOut);
    void blurPass(glm::mat4 projection, GLuint FBO, GLuint textureIn, GLuint textureOut, int direction);
    void shadingPass(glm::mat4 projection, GLuint textureIn);
    void thicknessPass(glm::mat4 projection, GLuint FBO, GLuint textureOut);
    void setupQuad();
    void updateBuffer();
    GLuint setupFBO(GLuint texture);
    GLuint setupTexture();
    Camera &camera;
    particleSystem &simulation;
    GLuint particleVAO, particleVBO, skyBoxVAO, skyBoxVBO, cubeMapTexture, quadVAO, quadVBO,
            firstFBO, secondFBO, thicknessTexture, textureOne, textureTwo;
    Shader depthShader, skyBoxShader, blurShader, shadingShader;
    int width, height;
    static const GLfloat skyBoxVertices[108];
    static const GLfloat quadVertices[18];
    std::string srcPath;
};


#endif //PBF_15418_GLRENDERER_H
