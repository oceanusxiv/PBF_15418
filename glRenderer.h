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


class glRenderer {

public:
    glRenderer(int width, int height, Camera& camera, std::vector<glm::vec3> &particles) :
            particleShader("shaders/triangle_vs.glsl", "shaders/triangle_fs.glsl"),
            skyBoxShader("shaders/skybox_vs.glsl", "shaders/skybox_fs.glsl"),
            width(width),
            height(height),
            camera(camera),
            particles(particles) {}
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
    static inline GLuint loadCubeMap(std::vector<const GLchar *> faces)
    {
        GLuint textureID;
        glGenTextures(1, &textureID);

        int width,height;
        unsigned char* image;

        glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
        for(GLuint i = 0; i < faces.size(); i++)
        {
            image = SOIL_load_image(faces[i], &width, &height, 0, SOIL_LOAD_RGB);
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
    void drawParticles(glm::mat4 projection);
    Camera &camera;
    std::vector<glm::vec3> &particles;
    GLuint particleVao, particleVbo, skyBoxVao, skyBoxVbo, cubeMapTexture;
    Shader particleShader, skyBoxShader;
    int width, height;
    static const GLfloat skyBoxVertices[108];
};


#endif //PBF_15418_GLRENDERER_H
