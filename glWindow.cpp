//
// Created by Eric Fang on 2/6/17.
//

#include "glWindow.h"

#ifdef __APPLE__
#define OSX true
#else
#define OSX false
#endif

int glWindow::init() {

    if (!glfwInit()) {
        std::cout << "Could not open GLFW!" << std::endl;
        return 1;
    }

    if (OSX) {
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    }

    window = glfwCreateWindow(width, height, "Hello Fluid", nullptr, nullptr);
    if (!window) {
        std::cout << "Could not open Window!" << std::endl;
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return 1;
    }

    const GLubyte *renderer = glGetString(GL_RENDERER);
    const GLubyte *version = glGetString(GL_VERSION);
    std::cout << "Renderer: " << renderer << std::endl;
    std::cout << "OpenGL version supported: " << version << std::endl;

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_DEPTH_CLAMP);

    glEnable(GL_PROGRAM_POINT_SIZE);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);

    glViewport(0, 0, w, h);


    glfwSetWindowUserPointer(window, this);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    return 0;
}

void glWindow::updateTime() {
    GLfloat  currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
}

// Moves/alters the camera positions based on user input
void glWindow::doMovement()
{
    // Camera controls
    if(keys[GLFW_KEY_W])
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if(keys[GLFW_KEY_S])
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if(keys[GLFW_KEY_A])
        camera.ProcessKeyboard(LEFT, deltaTime);
    if(keys[GLFW_KEY_D])
        camera.ProcessKeyboard(RIGHT, deltaTime);
}

void glWindow::keysRegistered(int key, int action) {

    //std::cout << key << std::endl;
    if (key >= 0 && key < 1024)
    {
        if(action == GLFW_PRESS)
            keys[key] = true;
        else if(action == GLFW_RELEASE)
            keys[key] = false;
    }
}

void glWindow::mouseRegistered(double xpos, double ypos) {
    if(firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    GLfloat xoffset = xpos - lastX;
    GLfloat yoffset = lastY - ypos;  // Reversed since y-coordinates go from bottom to left

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

void glWindow::scrollRegistered(double xoffset, double yoffset) {
    camera.ProcessMouseScroll(yoffset);
}

