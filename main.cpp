#include "glWindow.h"
#include "glRenderer.h"
#include "particleSystem.h"

int main() {

    particleSystem sim(150000, glm::vec3(1, 1, 1));

    glWindow simWindow(800, 600);
    simWindow.init();

    glRenderer simRenderer(800, 600, simWindow.getCamera(), sim.getParticles());
    simRenderer.init();

    while(!glfwWindowShouldClose(simWindow.getWindow()))
    {
        simWindow.updateTime();
        glfwPollEvents();
        simWindow.doMovement();

        simRenderer.onDraw();

        glfwSwapBuffers(simWindow.getWindow());
    }

    glfwTerminate();
    return 0;
}