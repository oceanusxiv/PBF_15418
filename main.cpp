
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "glWindow.h"
#include "glRenderer.h"
#include "particleSystem.h"

int main() {

    particleSystem sim(100000, glm::vec3(1, 0.1, 1));

    glWindow simWindow(800, 600);
    simWindow.init();

    glRenderer simRenderer(800, 600, simWindow.getCamera(), sim.getParticles());
    simRenderer.init();

    while(!glfwWindowShouldClose(simWindow.getWindow()))
    {
        simWindow.updateTime();
        glfwPollEvents();
        simWindow.doMovement();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        simRenderer.onDraw();

        glfwSwapBuffers(simWindow.getWindow());
    }

    glfwTerminate();
    return 0;
}