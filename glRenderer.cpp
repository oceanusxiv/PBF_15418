//
// Created by Eric Fang on 2/7/17.
//

#include "glRenderer.h"

const GLfloat glRenderer::skyBoxVertices[108] = {
        // Positions
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
        1.0f,  1.0f, -1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f,  1.0f
};

int glRenderer::init() {

    setupParticles();

    setupSkyBox();

    return 0;
}

void glRenderer::onDraw() {

    glm::mat4 projection = glm::perspective(camera.Zoom, (float)width/(float)height, 0.1f, 100.f);

    drawParticles(projection);
    drawSkyBox(projection);
}

void glRenderer::setupSkyBox() {

    glGenVertexArrays(1, &skyBoxVao);
    glGenBuffers(1, &skyBoxVbo);
    glBindVertexArray(skyBoxVao);
    glBindBuffer(GL_ARRAY_BUFFER, skyBoxVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyBoxVertices), &skyBoxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glBindVertexArray(0);

    std::vector<const GLchar*> faces;
    faces.push_back("skybox/right.jpg");
    faces.push_back("skybox/left.jpg");
    faces.push_back("skybox/top.jpg");
    faces.push_back("skybox/bottom.jpg");
    faces.push_back("skybox/back.jpg");
    faces.push_back("skybox/front.jpg");
    cubeMapTexture = loadCubeMap(faces);
}

void glRenderer::drawSkyBox(glm::mat4 projection) {
    // Draw skybox as last
    glDepthFunc(GL_LEQUAL);  // Change depth function so depth test passes when values are equal to depth buffer's content
    skyBoxShader.Use();
    glm::mat4 view = glm::mat4(glm::mat3(camera.GetViewMatrix()));	// Remove any translation component of the view matrix
    glUniformMatrix4fv(glGetUniformLocation(skyBoxShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(skyBoxShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    // skybox cube
    glBindVertexArray(skyBoxVao);
    glActiveTexture(GL_TEXTURE0);
    //glUniform1i(glGetUniformLocation(shader.Program, "skybox"), 0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTexture);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
    glDepthFunc(GL_LESS); // Set depth function back to default
}

void glRenderer::setupParticles() {

    glGenBuffers(1, &particleVbo);
    glBindBuffer(GL_ARRAY_BUFFER, particleVbo);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(glm::vec3), &particles[0].x, GL_STATIC_DRAW);

    glGenVertexArrays(1, &particleVao);
    glBindVertexArray(particleVao);

    glBindBuffer(GL_ARRAY_BUFFER, particleVbo);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void glRenderer::drawParticles(glm::mat4 projection) {

    particleShader.Use();

    // Create camera transformation
    glm::mat4 model;
    glm::mat4 view = camera.GetViewMatrix();

    GLfloat pointRadius = 0.02f;
    GLfloat pointScale = 1000.0f;

    // Pass the matrices to the shader
    glUniformMatrix4fv(glGetUniformLocation(particleShader.Program, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(particleShader.Program, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(particleShader.Program, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
    glUniform1f(glGetUniformLocation(particleShader.Program, "pointRadius"), pointRadius);
    glUniform1f(glGetUniformLocation(particleShader.Program, "pointScale"), pointScale);

    glBindVertexArray(particleVao);
    glDrawArrays(GL_POINTS, 0, particles.size());
    glBindVertexArray(0);
}

glRenderer::~glRenderer() {
    glDeleteVertexArrays(1, &particleVao);
    glDeleteVertexArrays(1, &skyBoxVao);
    glDeleteBuffers(1, &particleVbo);
    glDeleteBuffers(1, &skyBoxVbo);
}
