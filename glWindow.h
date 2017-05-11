//
// Created by Eric Fang on 2/6/17.
//

#ifndef PBF_15418_GLWINDOW_H
#define PBF_15418_GLWINDOW_H

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Camera.h"
#include <glm/glm.hpp>

class glWindow {

public:
    glWindow(int width, int height) :
            width(width),
            height(height),
            camera(glm::vec3(50.f, 150.f, -100.f)),
            firstMouse(true),
            lastX(width/2),
            lastY(height/2),
            deltaTime(0.0f),
            lastFrame(0.0f) {}
    int init();
    GLFWwindow *getWindow() { return window; }
    Camera &getCamera() { return camera; }
    void updateTime();
    void doMovement();

private:
    inline static void key_callback(GLFWwindow* win, int key, int scancode, int action, int mode) {
        if(key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(win, GL_TRUE);
        else {
            glWindow *window = static_cast<glWindow *>(glfwGetWindowUserPointer(win));
            window->keysRegistered(key, action);
        }
    }
    inline static void scroll_callback(GLFWwindow* win, double xoffset, double yoffset) {
        glWindow *window = static_cast<glWindow *>(glfwGetWindowUserPointer(win));
        window->scrollRegistered(xoffset, yoffset);
    }
    inline static void mouse_callback(GLFWwindow* win, double xpos, double ypos) {
        glWindow *window = static_cast<glWindow *>(glfwGetWindowUserPointer(win));
        window->mouseRegistered(xpos, ypos);
    }
    void keysRegistered(int key, int action);
    void scrollRegistered(double xoffset, double yoffset);
    void mouseRegistered(double xpos, double ypos);
    int width, height;
    GLFWwindow *window;
    Camera camera;
    bool keys[1024], firstMouse;
    GLfloat lastX, lastY, deltaTime, lastFrame;

};


#endif //PBF_15418_GLWINDOW_H
